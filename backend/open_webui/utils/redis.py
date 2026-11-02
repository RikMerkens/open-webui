"""Redis connection utilities.

Provides connection factory functions for standalone, Sentinel, and Cluster
Redis deployments, with optional async support and automatic connection caching.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from typing import Any
from urllib.parse import ParseResult, urlparse, urlunparse

import redis as _redis_sync
from open_webui.env import (
    REDIS_CLUSTER,
    REDIS_CLUSTER_REMAP_HOSTNAME,
    REDIS_ENABLE_AZURE_ENTRA_AUTH,
    REDIS_HEALTH_CHECK_INTERVAL,
    REDIS_RECONNECT_DELAY,
    REDIS_SENTINEL_HOSTS,
    REDIS_SENTINEL_MAX_RETRY_COUNT,
    REDIS_SENTINEL_PORT,
    REDIS_SOCKET_CONNECT_TIMEOUT,
    REDIS_SOCKET_KEEPALIVE,
    REDIS_SOCKET_TIMEOUT,
    REDIS_URL,
)

log = logging.getLogger(__name__)

_ACCEPTED_SCHEMES = frozenset({'redis', 'rediss'})
_SENTINEL_RETRYABLE = (
    _redis_sync.exceptions.ConnectionError,
    _redis_sync.exceptions.ReadOnlyError,
    _redis_sync.exceptions.TimeoutError,
)
_FACTORY_METHODS = frozenset({'pipeline', 'pubsub', 'monitor', 'client', 'transaction', 'lock'})
_CONNECTION_POOL: dict[tuple, Any] = {}


def parse_redis_url(url: str) -> dict[str, Any]:
    """Break a ``redis://`` URL into its parts: service, port, db, username, password."""
    parts: ParseResult = urlparse(url)
    if parts.scheme not in _ACCEPTED_SCHEMES:
        raise ValueError(f"Invalid Redis URL scheme '{parts.scheme}'; expected 'redis' or 'rediss'.")
    return {
        # hostname lowercases, but Sentinel service names are case-sensitive
        'service': parts.netloc.rpartition('@')[2].split(':')[0] or 'mymaster',
        'port': parts.port or 6379,
        'db': int(parts.path.lstrip('/') or '0'),
        'username': parts.username or None,
        'password': parts.password or None,
    }


parse_redis_service_url = parse_redis_url


def cluster_hostname_remap(public_host: str):
    """Replace a discovered cluster hostname and keep that shard's port.

    Azure Managed Redis CLUSTER SLOTS returns node IP addresses. The TLS certificate
    is issued for the public hostname. Each shard keeps the port the cluster
    reported (port 10000 is only the first connection point).
    """

    def remap(address: tuple[str, int]) -> tuple[str, int]:
        return public_host, address[1]

    return remap


def redis_options_without_credentials(options: dict[str, Any]) -> dict[str, Any]:
    """Drop username and password so they are not combined with a credential provider."""
    cleaned = dict(options)
    cleaned.pop('username', None)
    cleaned.pop('password', None)
    sentinel_kwargs = cleaned.get('sentinel_kwargs')
    if isinstance(sentinel_kwargs, dict):
        sentinel_kwargs = dict(sentinel_kwargs)
        sentinel_kwargs.pop('username', None)
        sentinel_kwargs.pop('password', None)
        cleaned['sentinel_kwargs'] = sentinel_kwargs
    return cleaned


def redis_url_for_entra(url: str) -> str:
    """Drop URL credentials when Azure Entra authentication supplies them."""
    if not url or not REDIS_ENABLE_AZURE_ENTRA_AUTH:
        return url
    parts = urlparse(url)
    if parts.username or parts.password:
        log.info('Ignoring username and password in the Redis URL because Azure Entra authentication is enabled')
    hostname = parts.hostname or ''
    host = f'[{hostname}]' if ':' in hostname else hostname
    netloc = f'{host}:{parts.port}' if parts.port else host
    return urlunparse((parts.scheme, netloc, parts.path, '', parts.query, ''))


def get_sentinels_from_env(
    hosts_csv: str | None,
    port: str | int | None,
) -> list[tuple[str, int]]:
    """Turn a comma-separated host string into ``[(host, port), …]``."""
    if not hosts_csv:
        return []
    resolved_port = int(port) if port else 26379
    return [(host.strip(), resolved_port) for host in hosts_csv.split(',') if host.strip()]


def build_sentinel_url(
    base_url: str,
    hosts_csv: str,
    port: str | int,
) -> str:
    """Construct a ``redis+sentinel://`` connection string.

    ``base_url`` supplies credentials, db index, and master service name.
    ``hosts_csv`` is a comma-separated list of sentinel hostnames.
    """
    cfg = parse_redis_url(base_url)
    auth = ''
    if cfg['username'] or cfg['password']:
        auth = f'{cfg["username"] or ""}:{cfg["password"] or ""}@'
    nodes = ','.join(f'{host.strip()}:{port}' for host in hosts_csv.split(',') if host.strip())
    return f'redis+sentinel://{auth}{nodes}/{cfg["db"]}/{cfg["service"]}'


def build_sentinel_kwargs(base_url: str, redis_options: dict[str, Any]) -> dict[str, Any]:
    """Kwargs for the Sentinel connections: ``base_url`` credentials plus the ``socket_*`` options."""
    cfg = parse_redis_url(base_url)
    socket_options = {key: value for key, value in redis_options.items() if key.startswith('socket_')}
    return {'username': cfg['username'], 'password': cfg['password'], **socket_options}


def get_redis_client(async_mode: bool = False) -> Any | None:
    """Create a Redis connection using settings from environment variables.

    Returns ``None`` when Redis is not configured or the connection fails.
    """
    sentinel_list = get_sentinels_from_env(REDIS_SENTINEL_HOSTS, REDIS_SENTINEL_PORT)
    if not REDIS_URL and not sentinel_list:
        return None
    try:
        return get_redis_connection(
            REDIS_URL,
            redis_sentinels=sentinel_list,
            redis_cluster=REDIS_CLUSTER,
            async_mode=async_mode,
        )
    except Exception:
        if REDIS_ENABLE_AZURE_ENTRA_AUTH:
            log.exception('Could not establish Redis connection with Azure Entra authentication')
        else:
            log.debug('Could not establish Redis connection', exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Sentinel proxy with automatic failover retry
# ---------------------------------------------------------------------------


class SentinelRedisProxy:
    """Transparent proxy that re-resolves the Sentinel master on connection errors.

    Every call (sync or async) is wrapped with retry logic so that transient
    Sentinel failovers are handled without caller intervention.
    """

    def __init__(
        self,
        sentinel: Any,
        service_name: str,
        *,
        async_mode: bool = True,
    ) -> None:
        self._sentinel = sentinel
        self._service_name = service_name
        self._async_mode = async_mode
        self._master: Any | None = None

    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access with automatic Sentinel failover retry."""
        current_master = self._resolve_master()
        original = getattr(current_master, name)

        # Non-callable or factory attributes pass through without wrapping.
        if not callable(original) or name in _FACTORY_METHODS:
            return original

        # Select the retry wrapper matching the execution mode.
        if not self._async_mode:
            return self._wrap_sync(name)
        return self._wrap_async(name, original)

    def _resolve_master(self) -> Any:
        """Ask Sentinel for the current master connection."""
        if self._master is None:
            self._master = self._sentinel.master_for(self._service_name)
        return self._master

    def _clear_master(self) -> None:
        self._master = None

    def _should_retry(self, attempt: int) -> bool:
        return attempt < REDIS_SENTINEL_MAX_RETRY_COUNT - 1

    def _log_retry(self, exc: Exception, attempt: int) -> None:
        log.debug(
            'Sentinel failover (%s) — retry %d/%d',
            type(exc).__name__,
            attempt + 1,
            REDIS_SENTINEL_MAX_RETRY_COUNT,
        )

    def _log_exhausted(self, exc: Exception) -> None:
        log.error(
            'Redis operation failed after %d retries: %s',
            REDIS_SENTINEL_MAX_RETRY_COUNT,
            exc,
        )

    # -- async wrappers -----------------------------------------------------

    def _wrap_async(self, name: str, attr: Any) -> Any:
        if inspect.isasyncgenfunction(attr):
            return self._wrap_async_gen(name)
        return self._wrap_async_call(name)

    def _wrap_async_gen(self, name: str) -> Any:
        proxy = self

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            async def _inner():
                for attempt in range(REDIS_SENTINEL_MAX_RETRY_COUNT):
                    try:
                        method = getattr(proxy._resolve_master(), name)
                        async for value in method(*args, **kwargs):
                            yield value
                        return
                    except _SENTINEL_RETRYABLE as exc:
                        if proxy._should_retry(attempt):
                            proxy._log_retry(exc, attempt)
                            proxy._clear_master()
                            if REDIS_RECONNECT_DELAY:
                                await asyncio.sleep(REDIS_RECONNECT_DELAY / 1000)
                            continue
                        proxy._log_exhausted(exc)
                        raise

            return _inner()

        return wrapper

    def _wrap_async_call(self, name: str) -> Any:
        proxy = self

        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            for attempt in range(REDIS_SENTINEL_MAX_RETRY_COUNT):
                try:
                    method = getattr(proxy._resolve_master(), name)
                    result = method(*args, **kwargs)
                    if inspect.iscoroutine(result):
                        return await result
                    return result
                except _SENTINEL_RETRYABLE as exc:
                    if proxy._should_retry(attempt):
                        proxy._log_retry(exc, attempt)
                        proxy._clear_master()
                        if REDIS_RECONNECT_DELAY:
                            await asyncio.sleep(REDIS_RECONNECT_DELAY / 1000)
                        continue
                    proxy._log_exhausted(exc)
                    raise

        return wrapper

    # -- sync wrapper -------------------------------------------------------

    def _wrap_sync(self, name: str) -> Any:
        proxy = self

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for attempt in range(REDIS_SENTINEL_MAX_RETRY_COUNT):
                try:
                    method = getattr(proxy._resolve_master(), name)
                    return method(*args, **kwargs)
                except _SENTINEL_RETRYABLE as exc:
                    if proxy._should_retry(attempt):
                        proxy._log_retry(exc, attempt)
                        proxy._clear_master()
                        if REDIS_RECONNECT_DELAY:
                            time.sleep(REDIS_RECONNECT_DELAY / 1000)
                        continue
                    proxy._log_exhausted(exc)
                    raise

        return wrapper


# ---------------------------------------------------------------------------
# Connection factory
# ---------------------------------------------------------------------------


def _socket_options() -> dict[str, Any]:
    """Collect optional socket-level kwargs once instead of repeating them."""
    opts: dict[str, Any] = {}
    if REDIS_SOCKET_CONNECT_TIMEOUT is not None:
        opts['socket_connect_timeout'] = REDIS_SOCKET_CONNECT_TIMEOUT
    if REDIS_SOCKET_TIMEOUT:
        opts['socket_timeout'] = REDIS_SOCKET_TIMEOUT
    if REDIS_SOCKET_KEEPALIVE:
        opts['socket_keepalive'] = True
    if REDIS_HEALTH_CHECK_INTERVAL:
        opts['health_check_interval'] = REDIS_HEALTH_CHECK_INTERVAL
    return opts


def _build_sentinel(
    redis_module: Any,
    url: str,
    sentinels: list[tuple[str, int]],
    decode_responses: bool,
    async_mode: bool,
    credential_provider: Any | None = None,
) -> SentinelRedisProxy:
    """Create a SentinelRedisProxy from a redis URL and sentinel list."""
    cfg = parse_redis_url(url)
    connection_kwargs: dict[str, Any] = {
        'port': cfg['port'],
        'db': cfg['db'],
        'decode_responses': decode_responses,
        'socket_connect_timeout': REDIS_SOCKET_CONNECT_TIMEOUT,
        **{k: v for k, v in _socket_options().items() if k != 'socket_connect_timeout'},
    }
    if credential_provider is not None:
        connection_kwargs['credential_provider'] = credential_provider
    else:
        connection_kwargs['username'] = cfg['username']
        connection_kwargs['password'] = cfg['password']
    sentinel = redis_module.sentinel.Sentinel(
        sentinels,
        sentinel_kwargs=build_sentinel_kwargs(
            url, {'socket_connect_timeout': REDIS_SOCKET_CONNECT_TIMEOUT, **_socket_options()}
        ),
        **connection_kwargs,
    )
    return SentinelRedisProxy(sentinel, cfg['service'], async_mode=async_mode)


def get_redis_connection(
    redis_url: str | None,
    redis_sentinels: list[tuple[str, int]] | None = None,
    redis_cluster: bool = False,
    async_mode: bool = False,
    decode_responses: bool = True,
) -> Any | None:
    """Return a cached Redis connection (or create one).

    Supports three topologies in order of precedence:
    1. **Sentinel** — when ``redis_sentinels`` is non-empty.
    2. **Cluster** — when ``redis_cluster`` is True.
    3. **Standalone** — plain ``redis://`` connection.
    """
    cache_key = (
        redis_url,
        tuple(redis_sentinels) if redis_sentinels else (),
        redis_cluster,
        async_mode,
        decode_responses,
    )
    if cache_key in _CONNECTION_POOL:
        return _CONNECTION_POOL[cache_key]

    extra = _socket_options()
    connection: Any = None
    from open_webui.utils.azure_credential import azure_redis_credential_provider

    provider = azure_redis_credential_provider()
    if provider is not None:
        extra['credential_provider'] = provider
    connect_url = redis_url_for_entra(redis_url) if provider is not None and redis_url else redis_url

    # Pick the right redis module for sync vs async.
    if async_mode:
        import redis.asyncio as redis_mod
    else:
        import redis as redis_mod  # type: ignore[no-redef]

    if redis_sentinels:
        connection = _build_sentinel(
            redis_mod,
            redis_url,
            redis_sentinels,
            decode_responses,
            async_mode,
            credential_provider=provider,
        )
    elif redis_cluster:
        if not connect_url:
            raise ValueError('Redis URL is required for cluster mode.')
        cluster_kwargs = dict(extra)
        if REDIS_CLUSTER_REMAP_HOSTNAME:
            parts = urlparse(connect_url)
            if parts.hostname:
                cluster_kwargs['address_remap'] = cluster_hostname_remap(parts.hostname)
            else:
                log.warning('REDIS_CLUSTER_REMAP_HOSTNAME is set, but the Redis URL has no hostname')
        connection = redis_mod.cluster.RedisCluster.from_url(
            connect_url,
            decode_responses=decode_responses,
            **cluster_kwargs,
        )
    elif connect_url:
        factory = getattr(redis_mod, 'from_url', None) or redis_mod.Redis.from_url
        # reconnect once when Redis has closed a pooled connection, e.g. after a restart
        retry = redis_mod.retry.Retry(_redis_sync.backoff.NoBackoff(), 1, supported_errors=(redis_mod.ConnectionError,))
        connection = factory(connect_url, decode_responses=decode_responses, retry=retry, **extra)

    _CONNECTION_POOL[cache_key] = connection
    return connection
