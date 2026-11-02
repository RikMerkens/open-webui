"""Microsoft Entra tokens for Azure Database for PostgreSQL and Azure Managed Redis.

Tokens are fetched with ``DefaultAzureCredential`` (managed identity on Azure,
Azure CLI when ``AZURE_TOKEN_CREDENTIALS=AzureCliCredential``). A user-assigned
managed identity is selected with ``AZURE_CLIENT_ID``.

The cache is shared by PostgreSQL and Redis. A fetch does not hold that lock, and
callers on the asyncio event loop never make the network call themselves. A
background thread refreshes each scope five minutes before expiry, and at least
30 seconds apart when a token is already inside that window.
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import logging
import threading
import time
import weakref
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from redis.credentials import StreamingCredentialProvider

log = logging.getLogger(__name__)

POSTGRES_TOKEN_SCOPE = 'https://ossrdbms-aad.database.windows.net/.default'
REDIS_TOKEN_SCOPE = 'https://redis.azure.com/.default'
_REFRESH_SKEW = timedelta(minutes=5)
_MIN_REFRESH_WAIT_SECONDS = 30

_credential = None
_credential_lock = threading.Lock()
_token_lock = threading.Lock()
_token_cache: dict[str, tuple[AzureAccessToken, datetime]] = {}
_refresh_scopes: set[str] = set()
_refresh_wakeup = threading.Event()
_refresh_stop = threading.Event()
_refresh_thread: threading.Thread | None = None
_refresh_thread_lock = threading.Lock()


@dataclass(frozen=True)
class AzureAccessToken:
    token: str
    expires_on: int


@dataclass
class _Listener:
    """One redis-py re-auth callback, weakly held, with the loop that owns it."""

    callback: weakref.ref
    loop: asyncio.AbstractEventLoop | None
    key: tuple[int, int]


def get_azure_credential():
    """Return the process-wide ``DefaultAzureCredential``."""
    global _credential
    if _credential is None:
        with _credential_lock:
            if _credential is None:
                from azure.identity import DefaultAzureCredential

                _credential = DefaultAzureCredential()
    return _credential


def refresh_wait_seconds(expires_on: int, now: float | None = None) -> float:
    """Seconds until a token should be refreshed, never less than 30."""
    current = time.time() if now is None else now
    return max(expires_on - current - _REFRESH_SKEW.total_seconds(), _MIN_REFRESH_WAIT_SECONDS)


def _on_event_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def _read_cached_token(scope: str, now: datetime, *, allow_until_expiry: bool) -> AzureAccessToken | None:
    cached = _token_cache.get(scope)
    if cached is None:
        return None
    token, refresh_at = cached
    if now < refresh_at:
        return token
    if allow_until_expiry and now.timestamp() < token.expires_on:
        return token
    return None


def get_azure_access_token(scope: str, *, force: bool = False) -> AzureAccessToken:
    """Return a cached Entra access token for ``scope``.

    The cache is discarded five minutes before ``expires_on``. The token value is
    never logged. On the event loop this returns a cached token or raises; it does
    not call Azure.
    """
    now = datetime.now(timezone.utc)
    if not force:
        with _token_lock:
            cached = _read_cached_token(scope, now, allow_until_expiry=False)
            if cached is not None:
                return cached

    if _on_event_loop():
        ensure_azure_token_refresh(scope)
        with _token_lock:
            cached = _read_cached_token(scope, now, allow_until_expiry=True)
            if cached is not None:
                return cached
        raise RuntimeError(
            f'Azure Entra token for {scope} is not cached. '
            'Refusing to call Azure from the event loop; the background refresh has not produced a token yet.'
        )

    token = fetch_azure_access_token(scope)
    ensure_azure_token_refresh(scope)
    return token


def fetch_azure_access_token(scope: str) -> AzureAccessToken:
    """Fetch a token and store it. The cache lock is not held during the network call."""
    access = get_azure_credential().get_token(scope)
    expires_on = int(access.expires_on)
    now = datetime.now(timezone.utc)
    refresh_at = datetime.fromtimestamp(expires_on, tz=timezone.utc) - _REFRESH_SKEW
    if refresh_at <= now:
        refresh_at = now + timedelta(seconds=_MIN_REFRESH_WAIT_SECONDS)
    token = AzureAccessToken(token=access.token, expires_on=expires_on)
    with _token_lock:
        current = _token_cache.get(scope)
        if current is not None and current[0].expires_on > expires_on and now < current[1]:
            return current[0]
        _token_cache[scope] = (token, refresh_at)
    log.info('Azure Entra token refreshed for %s; refresh after %s', scope, refresh_at.isoformat())
    return token


def ensure_azure_token_refresh(scope: str) -> None:
    """Keep ``scope`` refreshed on the background thread."""
    global _refresh_thread
    with _refresh_thread_lock:
        _refresh_scopes.add(scope)
        if _refresh_thread is None or not _refresh_thread.is_alive():
            _refresh_stop.clear()
            _refresh_thread = threading.Thread(target=_scope_refresh_loop, name='azure-entra-token', daemon=True)
            _refresh_thread.start()
    _refresh_wakeup.set()


def _refresh_plan(now: datetime) -> tuple[list[str], float]:
    with _refresh_thread_lock:
        scopes = list(_refresh_scopes)
    due: list[str] = []
    soonest: float | None = None
    with _token_lock:
        for scope in scopes:
            cached = _token_cache.get(scope)
            if cached is None or now >= cached[1]:
                due.append(scope)
                continue
            remaining = (cached[1] - now).total_seconds()
            soonest = remaining if soonest is None else min(soonest, remaining)
    if soonest is None or soonest <= 0:
        return due, _MIN_REFRESH_WAIT_SECONDS
    return due, soonest


def _scope_refresh_loop() -> None:
    while not _refresh_stop.is_set():
        _refresh_wakeup.clear()
        due, delay = _refresh_plan(datetime.now(timezone.utc))
        failed = False
        for scope in due:
            try:
                fetch_azure_access_token(scope)
            except Exception:
                log.exception('Azure Entra token refresh failed for %s', scope)
                failed = True
        if due:
            _, delay = _refresh_plan(datetime.now(timezone.utc))
        if failed or delay <= 0:
            delay = _MIN_REFRESH_WAIT_SECONDS
        if _wait_for_refresh(delay):
            return


def _wait_for_refresh(delay: float) -> bool:
    """Return True when the refresh thread should exit."""
    if _refresh_stop.is_set():
        return True
    deadline = time.time() + delay
    while not _refresh_stop.is_set():
        remaining = deadline - time.time()
        if remaining <= 0:
            return False
        if _refresh_wakeup.wait(remaining):
            _refresh_wakeup.clear()
            return _refresh_stop.is_set()
    return True


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


def _weak_callback(callback: Callable[..., Any]) -> weakref.ref:
    if inspect.ismethod(callback):
        return weakref.WeakMethod(callback)
    return weakref.ref(callback)


def _callback_key(callback: Callable[..., Any]) -> tuple[int, int]:
    owner = getattr(callback, '__self__', None)
    func = getattr(callback, '__func__', callback)
    return (id(owner) if owner is not None else id(callback), id(func))


def _running_loop() -> asyncio.AbstractEventLoop | None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return None
    if loop.is_running():
        return loop
    return None


class AzureRedisCredentialProvider(StreamingCredentialProvider):
    """Refresh an Entra token and re-authenticate pooled Redis connections.

    redis-py calls ``on_next`` when a client is created. Later refreshes invoke
    that callback with a token whose ``oid`` claim is the Redis username.
    Callbacks are held weakly, and each one keeps the event loop that registered
    it, so a closed client can be collected and is not authenticated again.
    """

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._callbacks: list[_Listener] = []
        self._error_callbacks: list[_Listener] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._token: _RedisAccessToken | None = None

    def is_streaming(self) -> bool:
        return True

    def on_next(self, callback: Callable[..., Any]) -> None:
        self._add_listener(self._callbacks, callback)
        self._ensure_refresh_thread()

    def on_error(self, callback: Callable[..., Any]) -> None:
        self._add_listener(self._error_callbacks, callback)

    def get_credentials(self) -> tuple[str, str]:
        token = self._current_token()
        object_id = token.try_get('oid')
        if not object_id:
            raise ValueError('Azure Redis Entra token has no oid claim')
        return object_id, token.get_value()

    async def get_credentials_async(self) -> tuple[str, str]:
        loop = asyncio.get_running_loop()
        with self._lock:
            for listener in self._callbacks + self._error_callbacks:
                if listener.loop is None:
                    listener.loop = loop
        return await asyncio.to_thread(self.get_credentials)

    def _add_listener(self, bucket: list[_Listener], callback: Callable[..., Any]) -> None:
        listener = _Listener(_weak_callback(callback), _running_loop(), _callback_key(callback))
        with self._lock:
            self._prune(bucket)
            for existing in bucket:
                if existing.key == listener.key:
                    if listener.loop is not None:
                        existing.loop = listener.loop
                    return
            bucket.append(listener)

    def _current_token(self) -> _RedisAccessToken:
        with self._lock:
            cached = self._token
        if cached is not None and time.time() < cached.expires_on - _REFRESH_SKEW.total_seconds():
            return cached
        access = get_azure_access_token(REDIS_TOKEN_SCOPE)
        token = _RedisAccessToken(access, jwt_claim(access.token, 'oid'))
        with self._lock:
            self._token = token
        return token

    def _ensure_refresh_thread(self) -> None:
        with self._lock:
            if self._thread is not None:
                return
            thread = threading.Thread(target=self._refresh_loop, name='azure-redis-entra', daemon=True)
            self._thread = thread
        thread.start()

    def _refresh_loop(self) -> None:
        notified_expires: int | None = None
        while not self._stop.is_set():
            try:
                token = self._current_token()
            except Exception:
                log.exception('Azure Redis Entra token could not be read')
                self._notify_errors()
                if self._stop.wait(_MIN_REFRESH_WAIT_SECONDS):
                    return
                continue
            if notified_expires is not None and token.expires_on != notified_expires:
                self._notify(token)
                log.info(
                    'Azure Redis Entra token refreshed; expires at %s',
                    datetime.fromtimestamp(token.expires_on, tz=timezone.utc).isoformat(),
                )
            notified_expires = token.expires_on
            if self._stop.wait(refresh_wait_seconds(token.expires_on)):
                return

    def _notify(self, token: _RedisAccessToken) -> None:
        for listener in self._snapshot(self._callbacks):
            callback = listener.callback()
            if callback is None:
                continue
            try:
                result = callback(token)
            except Exception:
                log.exception('Azure Redis re-authentication callback failed')
                continue
            self._schedule(listener, result, _log_async_reauth)

    def _notify_errors(self) -> None:
        error = RuntimeError('Azure Redis Entra token refresh failed')
        for listener in self._snapshot(self._error_callbacks):
            callback = listener.callback()
            if callback is None:
                continue
            try:
                result = callback(error)
            except Exception as exc:
                # redis-py's sync handler raises on purpose. One warning is enough.
                log.warning('Azure Redis Entra token refresh failed: %s', exc)
                continue
            self._schedule(listener, result, _log_async_error)

    def _schedule(self, listener: _Listener, result: Any, done: Callable[[asyncio.Future], None]) -> None:
        if not inspect.isawaitable(result):
            return
        loop = listener.loop
        if loop is None or not loop.is_running():
            result.close()
            log.error('Azure Redis re-authentication has no running event loop')
            return
        future = asyncio.run_coroutine_threadsafe(result, loop)
        future.add_done_callback(done)

    def _snapshot(self, bucket: list[_Listener]) -> list[_Listener]:
        with self._lock:
            self._prune(bucket)
            return list(bucket)

    def _prune(self, bucket: list[_Listener]) -> None:
        bucket[:] = [listener for listener in bucket if listener.callback() is not None]


def _log_async_reauth(future: asyncio.Future) -> None:
    try:
        future.result()
    except Exception:
        log.exception('Azure Redis asynchronous re-authentication failed')


def _log_async_error(future: asyncio.Future) -> None:
    try:
        future.result()
    except Exception as exc:
        log.warning('Azure Redis asynchronous token error callback failed: %s', exc)


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
            ensure_azure_token_refresh(REDIS_TOKEN_SCOPE)
        return _redis_provider
