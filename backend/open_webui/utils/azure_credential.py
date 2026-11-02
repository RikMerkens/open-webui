"""Microsoft Entra tokens for Azure Database for PostgreSQL and Azure Managed Redis.

Tokens are fetched with ``DefaultAzureCredential`` (managed identity on Azure,
Azure CLI when ``AZURE_TOKEN_CREDENTIALS=AzureCliCredential``). A user-assigned
managed identity is selected with ``AZURE_CLIENT_ID``.
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from redis.credentials import StreamingCredentialProvider

log = logging.getLogger(__name__)

POSTGRES_TOKEN_SCOPE = 'https://ossrdbms-aad.database.windows.net/.default'
REDIS_TOKEN_SCOPE = 'https://redis.azure.com/.default'
_REFRESH_SKEW = timedelta(minutes=5)

_credential = None
_credential_lock = threading.Lock()
_token_lock = threading.Lock()
_token_cache: dict[str, tuple[AzureAccessToken, datetime]] = {}


@dataclass(frozen=True)
class AzureAccessToken:
    token: str
    expires_on: int


def get_azure_credential():
    """Return the process-wide ``DefaultAzureCredential``."""
    global _credential
    if _credential is None:
        with _credential_lock:
            if _credential is None:
                from azure.identity import DefaultAzureCredential

                _credential = DefaultAzureCredential()
    return _credential


def get_azure_access_token(scope: str, *, force: bool = False) -> AzureAccessToken:
    """Return a cached Entra access token for ``scope``.

    The cache is discarded five minutes before ``expires_on`` so callers refresh
    before Azure rejects the token. The token value is never logged.
    """
    now = datetime.now(timezone.utc)
    with _token_lock:
        cached = _token_cache.get(scope)
        if not force and cached is not None and now < cached[1]:
            return cached[0]

        access = get_azure_credential().get_token(scope)
        expires_on = int(access.expires_on)
        refresh_at = datetime.fromtimestamp(expires_on, tz=timezone.utc) - _REFRESH_SKEW
        if refresh_at <= now:
            refresh_at = now + timedelta(seconds=30)
        token = AzureAccessToken(token=access.token, expires_on=expires_on)
        _token_cache[scope] = (token, refresh_at)
        log.info(
            'Azure Entra token refreshed for %s; refresh after %s',
            scope,
            refresh_at.isoformat(),
        )
        return token


def jwt_claim(token: str, claim: str) -> str:
    """Read one claim from a JWT payload without verifying the signature."""
    try:
        payload = token.split('.', 2)[1]
        padding = '=' * (-len(payload) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload + padding))
    except (IndexError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError('Azure access token payload could not be read') from exc
    value = claims.get(claim)
    if not value:
        raise ValueError(f'Azure access token has no {claim} claim')
    return str(value)


class _RedisAccessToken:
    """Duck-typed ``redis.auth.token.TokenInterface`` for pool re-authentication."""

    def __init__(self, access: AzureAccessToken, object_id: str) -> None:
        self._access = access
        self._object_id = object_id

    def try_get(self, key: str) -> str | None:
        if key == 'oid':
            return self._object_id
        return None

    def get_value(self) -> str:
        return self._access.token

    @property
    def expires_on(self) -> int:
        return self._access.expires_on


class AzureRedisCredentialProvider(StreamingCredentialProvider):
    """Refresh an Entra token and re-authenticate pooled Redis connections.

    redis-py calls ``on_next`` when a client is created. Later refreshes invoke
    that callback with a token whose ``oid`` claim is the Redis username.
    """

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._callbacks: list[Callable[..., Any]] = []
        self._error_callbacks: list[Callable[..., Any]] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._token: _RedisAccessToken | None = None

    def is_streaming(self) -> bool:
        return True

    def on_next(self, callback: Callable[..., Any]) -> None:
        with self._lock:
            self._callbacks.append(callback)
            self._capture_loop()
        self._ensure_refresh_thread()

    def on_error(self, callback: Callable[..., Any]) -> None:
        with self._lock:
            self._error_callbacks.append(callback)

    def get_credentials(self) -> tuple[str, str]:
        token = self._current_token()
        object_id = token.try_get('oid')
        if not object_id:
            raise ValueError('Azure Redis Entra token has no oid claim')
        return object_id, token.get_value()

    async def get_credentials_async(self) -> tuple[str, str]:
        with self._lock:
            self._capture_loop()
        return self.get_credentials()

    def _capture_loop(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if loop.is_running():
            self._loop = loop

    def _current_token(self, *, force: bool = False) -> _RedisAccessToken:
        with self._lock:
            if (
                not force
                and self._token is not None
                and time.time() < self._token.expires_on - _REFRESH_SKEW.total_seconds()
            ):
                return self._token
        access = get_azure_access_token(REDIS_TOKEN_SCOPE, force=force)
        token = _RedisAccessToken(access, jwt_claim(access.token, 'oid'))
        with self._lock:
            self._token = token
        return token

    def _ensure_refresh_thread(self) -> None:
        with self._lock:
            if self._thread is not None:
                return
            thread = threading.Thread(
                target=self._refresh_loop,
                name='azure-redis-entra',
                daemon=True,
            )
            self._thread = thread
        thread.start()

    def _refresh_loop(self) -> None:
        while not self._stop.is_set():
            try:
                token = self._current_token()
                delay = max(token.expires_on - time.time() - _REFRESH_SKEW.total_seconds(), 1)
            except Exception:
                log.exception('Azure Redis Entra token could not be read')
                self._notify_errors()
                if self._stop.wait(30):
                    return
                continue
            if self._stop.wait(delay):
                return
            try:
                token = self._current_token(force=True)
                self._notify(token)
                log.info(
                    'Azure Redis Entra token refreshed; expires at %s',
                    datetime.fromtimestamp(token.expires_on, tz=timezone.utc).isoformat(),
                )
            except Exception:
                log.exception('Azure Redis Entra token refresh failed')
                self._notify_errors()

    def _notify(self, token: _RedisAccessToken) -> None:
        with self._lock:
            callbacks = list(self._callbacks)
            loop = self._loop
        for callback in callbacks:
            try:
                result = callback(token)
            except Exception:
                log.exception('Azure Redis re-authentication callback failed')
                continue
            if not inspect.isawaitable(result):
                continue
            if loop is None or not loop.is_running():
                result.close()
                log.error('Azure Redis re-authentication has no running event loop')
                continue
            future = asyncio.run_coroutine_threadsafe(result, loop)
            future.add_done_callback(_log_async_reauth)

    def _notify_errors(self) -> None:
        error = RuntimeError('Azure Redis Entra token refresh failed')
        with self._lock:
            callbacks = list(self._error_callbacks)
            loop = self._loop
        for callback in callbacks:
            try:
                result = callback(error)
            except Exception:
                log.exception('Azure Redis token error callback failed')
                continue
            if inspect.isawaitable(result):
                if loop is None or not loop.is_running():
                    result.close()
                    continue
                asyncio.run_coroutine_threadsafe(result, loop)


def _log_async_reauth(future: asyncio.Future) -> None:
    try:
        future.result()
    except Exception:
        log.exception('Azure Redis asynchronous re-authentication failed')


_redis_provider: AzureRedisCredentialProvider | None = None
_redis_provider_lock = threading.Lock()


def azure_redis_credential_provider() -> AzureRedisCredentialProvider | None:
    """Return the shared Redis provider when Entra auth is enabled."""
    from open_webui.env import REDIS_ENABLE_AZURE_ENTRA_AUTH

    if not REDIS_ENABLE_AZURE_ENTRA_AUTH:
        return None
    global _redis_provider
    with _redis_provider_lock:
        if _redis_provider is None:
            _redis_provider = AzureRedisCredentialProvider()
        return _redis_provider
