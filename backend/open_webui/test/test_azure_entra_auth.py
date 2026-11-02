"""Unit tests for Azure Entra database and Redis authentication."""

from __future__ import annotations

import asyncio
import gc
import os
import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

_TEST_DIR = tempfile.mkdtemp(prefix='owui-entra-test-')
os.environ['DATA_DIR'] = _TEST_DIR
os.environ['WEBUI_SECRET_KEY'] = 'test-secret-key-for-entra-unit-tests'
os.environ['DATABASE_URL'] = f'sqlite:///{_TEST_DIR}/webui.db'
os.environ['DATABASE_ENABLE_AZURE_ENTRA_AUTH'] = 'false'
os.environ['DATABASE_ENABLE_IAM_TOKEN_AUTH'] = 'false'
os.environ['REDIS_ENABLE_AZURE_ENTRA_AUTH'] = 'false'
os.environ['REDIS_CLUSTER_REMAP_HOSTNAME'] = 'false'

import pytest
from open_webui.internal.db import AzureEntraTokenAuth
from open_webui.internal.db import enable_iam_token_auth
from open_webui.internal.db import event as db_event
from open_webui.utils import azure_credential, redis as redis_utils
from open_webui.utils.azure_credential import (
    AzureRedisCredentialProvider,
    _RedisAccessToken,
    jwt_claim,
    refresh_wait_seconds,
)
from open_webui.utils.redis import cluster_hostname_remap
from sqlalchemy.engine.url import make_url

import open_webui.internal.db as db


def _jwt(payload: dict) -> str:
    import base64
    import json

    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip('=')
    return f'header.{body}.sig'


def _token(value: str, expires_on: int, object_id: str = 'oid-1') -> _RedisAccessToken:
    return _RedisAccessToken(azure_credential.AzureAccessToken(value, expires_on), object_id)


class _Clock:
    def __init__(self, stops_after: int) -> None:
        self.delays: list[float] = []
        self._stops_after = stops_after

    def is_set(self) -> bool:
        return False

    def wait(self, delay: float) -> bool:
        self.delays.append(delay)
        return len(self.delays) >= self._stops_after


class _Client:
    def __init__(self) -> None:
        self.tokens: list[str] = []
        self.errors: list[BaseException] = []

    def reauth(self, token) -> None:
        self.tokens.append(token.get_value())

    def fail(self, error: BaseException) -> None:
        self.errors.append(error)
        raise error


@pytest.fixture(autouse=True)
def _clear_token_cache():
    azure_credential._token_cache.clear()
    azure_credential._refresh_scopes.clear()
    azure_credential._credential = None
    yield
    azure_credential._refresh_stop.set()
    azure_credential._refresh_wakeup.set()
    thread = azure_credential._refresh_thread
    if thread is not None and thread is not threading.current_thread():
        thread.join(timeout=1)
    azure_credential._refresh_thread = None
    azure_credential._refresh_stop.clear()
    azure_credential._refresh_wakeup.clear()
    azure_credential._token_cache.clear()
    redis_utils._CONNECTION_POOL.clear()


def test_cluster_remap_keeps_each_shard_port():
    remap = cluster_hostname_remap('cache.example.net')

    assert remap(('20.238.181.101', 8500)) == ('cache.example.net', 8500)
    assert remap(('20.238.181.101', 8501)) == ('cache.example.net', 8501)
    assert remap(('10.0.0.8', 10000)) == ('cache.example.net', 10000)


def test_cluster_remap_is_a_separate_setting(monkeypatch):
    captured: dict = {}

    def from_url(url, decode_responses=True, **kwargs):
        captured['url'] = url
        captured['kwargs'] = kwargs
        return object()

    monkeypatch.setattr(redis_utils, 'REDIS_CLUSTER_REMAP_HOSTNAME', True)
    monkeypatch.setattr(redis_utils, 'REDIS_ENABLE_AZURE_ENTRA_AUTH', False)
    monkeypatch.setattr('redis.cluster.RedisCluster.from_url', from_url)

    redis_utils.get_redis_connection(
        'rediss://cache.example.net:10000/0',
        redis_cluster=True,
        async_mode=False,
    )

    remap = captured['kwargs']['address_remap']
    assert remap(('10.0.0.1', 8500)) == ('cache.example.net', 8500)
    assert 'credential_provider' not in captured['kwargs']

    captured.clear()
    monkeypatch.setattr(redis_utils, 'REDIS_CLUSTER_REMAP_HOSTNAME', False)
    redis_utils._CONNECTION_POOL.clear()
    redis_utils.get_redis_connection(
        'rediss://cache.example.net:10000/0',
        redis_cluster=True,
        async_mode=False,
    )
    assert 'address_remap' not in captured['kwargs']


def test_redis_url_for_entra_strips_credentials_and_keeps_the_rest(monkeypatch):
    monkeypatch.setattr(redis_utils, 'REDIS_ENABLE_AZURE_ENTRA_AUTH', True)

    assert (
        redis_utils.redis_url_for_entra('rediss://user:secret@cache.example:10000/0?ssl_cert_reqs=required')
        == 'rediss://cache.example:10000/0?ssl_cert_reqs=required'
    )
    assert (
        redis_utils.redis_url_for_entra('rediss://user:secret@[2001:db8::1]:10000/0')
        == 'rediss://[2001:db8::1]:10000/0'
    )
    assert redis_utils.redis_url_for_entra('rediss://user:secret@cache.example/0') == 'rediss://cache.example/0'
    assert redis_utils.redis_url_for_entra('rediss://cache.example:10000/0') == 'rediss://cache.example:10000/0'


def test_sentinel_url_and_options_drop_credentials_before_the_provider(monkeypatch):
    monkeypatch.setattr(redis_utils, 'REDIS_ENABLE_AZURE_ENTRA_AUTH', True)
    source = redis_utils.redis_url_for_entra('rediss://user:secret@cache.example:10000/0')
    sentinel = redis_utils.build_sentinel_url(source, 's1,s2', 26379)
    options = redis_utils.redis_options_without_credentials(
        {
            'username': 'user',
            'password': 'secret',
            'sentinel_kwargs': redis_utils.build_sentinel_kwargs(
                'rediss://user:secret@cache.example:10000/0',
                {},
            ),
        }
    )

    assert 'secret' not in sentinel
    assert 'user:secret' not in sentinel
    assert 'username' not in options
    assert 'password' not in options
    assert 'username' not in options['sentinel_kwargs']
    assert 'password' not in options['sentinel_kwargs']


def test_jwt_claim_reads_oid_and_rejects_bad_tokens():
    assert jwt_claim(_jwt({'oid': '411f0ef5'}), 'oid') == '411f0ef5'

    with pytest.raises(ValueError, match='could not be read'):
        jwt_claim('not-a-jwt', 'oid')
    with pytest.raises(ValueError, match='could not be read'):
        jwt_claim('a.!!!.c', 'oid')
    with pytest.raises(ValueError, match='no oid claim'):
        jwt_claim(_jwt({'sub': 'someone'}), 'oid')


def test_cached_token_is_reused_and_fetch_does_not_hold_the_lock(monkeypatch):
    calls = {'n': 0}
    started = threading.Event()
    release = threading.Event()

    class Cred:
        def get_token(self, scope):
            calls['n'] += 1
            if scope == 'slow':
                started.set()
                assert release.wait(2)
            return SimpleNamespace(token=scope, expires_on=int(time.time()) + 3600)

    monkeypatch.setattr(azure_credential, 'get_azure_credential', lambda: Cred())
    monkeypatch.setattr(azure_credential, 'ensure_azure_token_refresh', lambda scope: None)

    first = azure_credential.get_azure_access_token('postgres')
    second = azure_credential.get_azure_access_token('postgres')
    assert first.token == second.token == 'postgres'
    assert calls['n'] == 1

    worker = threading.Thread(target=lambda: azure_credential.get_azure_access_token('slow'))
    worker.start()
    assert started.wait(2)
    began = time.time()
    fast = azure_credential.get_azure_access_token('fast')
    elapsed = time.time() - began
    release.set()
    worker.join(2)

    assert fast.token == 'fast'
    assert elapsed < 0.5


def test_event_loop_does_not_fetch(monkeypatch):
    monkeypatch.setattr(azure_credential, 'ensure_azure_token_refresh', lambda scope: None)

    def fail(scope):
        raise AssertionError(f'fetched {scope} on the event loop')

    monkeypatch.setattr(azure_credential, 'fetch_azure_access_token', fail)
    azure_credential._token_cache['loop-scope'] = (
        azure_credential.AzureAccessToken('still-valid', int(time.time()) + 120),
        datetime.now(timezone.utc) - timedelta(seconds=1),
    )

    async def cached():
        return azure_credential.get_azure_access_token('loop-scope')

    assert asyncio.run(cached()).token == 'still-valid'

    async def missing():
        azure_credential.get_azure_access_token('missing')

    with pytest.raises(RuntimeError, match='event loop'):
        asyncio.run(missing())


def test_refresh_wait_has_a_30_second_floor():
    now = 1_000_000.0
    assert refresh_wait_seconds(int(now + 60), now) == 30
    assert refresh_wait_seconds(int(now + 3600), now) == 3300


def test_refresh_loop_waits_and_reauthenticates_once_per_new_token():
    provider = AzureRedisCredentialProvider()
    provider._thread = threading.current_thread()
    short = _token('old', int(time.time()) + 60)
    fresh = _token('new', int(time.time()) + 3600)
    pending = iter((short, fresh))
    provider._current_token = lambda: next(pending)
    provider._stop = _Clock(stops_after=2)
    client = _Client()
    provider.on_next(client.reauth)

    provider._refresh_loop()

    assert provider._stop.delays[0] == 30
    assert provider._stop.delays[1] == pytest.approx(3300, abs=2)
    assert client.tokens == ['new']


def test_refresh_loop_error_waits_30_seconds_and_warns(caplog):
    provider = AzureRedisCredentialProvider()
    provider._thread = threading.current_thread()

    def boom():
        raise RuntimeError('credential down')

    provider._current_token = boom
    provider._stop = _Clock(stops_after=1)
    client = _Client()
    provider.on_error(client.fail)

    with caplog.at_level('WARNING'):
        provider._refresh_loop()

    assert provider._stop.delays == [30]
    assert client.errors
    assert 'Azure Redis Entra token refresh failed' in caplog.text
    assert 'Traceback' not in caplog.text.split('Azure Redis Entra token refresh failed')[-1]


def test_closed_client_is_forgotten_and_duplicate_registration_is_ignored():
    provider = AzureRedisCredentialProvider()
    provider._thread = threading.current_thread()
    client = _Client()
    provider.on_next(client.reauth)
    provider.on_next(client.reauth)
    assert len(provider._callbacks) == 1

    del client
    gc.collect()
    assert provider._snapshot(provider._callbacks) == []


def test_callbacks_keep_the_loop_that_registered_them():
    provider = AzureRedisCredentialProvider()
    provider._thread = threading.current_thread()
    ready = threading.Event()
    release = threading.Event()
    holder: dict = {}

    def other_loop() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        holder['loop'] = loop
        client = _Client()
        holder['client'] = client

        async def register():
            provider.on_next(client.reauth)

        loop.run_until_complete(register())
        ready.set()
        release.wait(2)
        loop.close()

    thread = threading.Thread(target=other_loop)
    thread.start()
    assert ready.wait(2)
    local = _Client()

    async def register_local():
        provider.on_next(local.reauth)

    asyncio.run(register_local())
    loops = [listener.loop for listener in provider._callbacks]
    assert holder['loop'] in loops
    assert any(loop is not holder['loop'] for loop in loops)
    release.set()
    thread.join(2)


def test_async_credentials_are_read_off_the_event_loop():
    provider = AzureRedisCredentialProvider()
    provider._thread = threading.current_thread()
    seen: dict = {}

    def current():
        seen['thread'] = threading.current_thread()
        return _token('tok', int(time.time()) + 3600)

    provider._current_token = current

    async def run():
        return await provider.get_credentials_async()

    assert asyncio.run(run()) == ('oid-1', 'tok')
    assert seen['thread'] is not threading.current_thread()


def test_database_token_auth_validates_the_url_and_reuses_the_cache(monkeypatch):
    with pytest.raises(ValueError, match='PostgreSQL'):
        AzureEntraTokenAuth('mysql://user@host/db')
    with pytest.raises(ValueError, match='host and user'):
        AzureEntraTokenAuth('postgresql:///openwebui')

    calls = {'n': 0}

    class Cred:
        def get_token(self, scope):
            calls['n'] += 1
            assert scope == azure_credential.POSTGRES_TOKEN_SCOPE
            return SimpleNamespace(token='db-token', expires_on=int(time.time()) + 3600)

    monkeypatch.setattr(azure_credential, 'get_azure_credential', lambda: Cred())
    monkeypatch.setattr(azure_credential, 'ensure_azure_token_refresh', lambda scope: None)
    auth = AzureEntraTokenAuth('postgresql://entra-user@db.example:5432/openwebui')

    assert auth.get_password() == 'db-token'
    assert auth.get_password() == 'db-token'
    assert calls['n'] == 1


def test_database_token_is_applied_only_when_host_port_and_user_match(monkeypatch, caplog):
    auth = AzureEntraTokenAuth('postgresql://entra-user@db.example/openwebui')
    monkeypatch.setattr(db, '_rds_iam_token_auth', None)
    monkeypatch.setattr(db, '_azure_entra_token_auth', auth)
    listened: list = []
    monkeypatch.setattr(db_event, 'contains', lambda *args, **kwargs: False)
    monkeypatch.setattr(db_event, 'listen', lambda engine, name, fn: listened.append(engine))

    matching = SimpleNamespace(url=make_url('postgresql://entra-user@db.example:5432/openwebui'))
    enable_iam_token_auth(matching)
    assert listened == [matching]

    other = SimpleNamespace(url=make_url('postgresql://other@elsewhere:5432/openwebui'))
    with caplog.at_level('WARNING'):
        enable_iam_token_auth(other)
    assert listened == [matching]
    assert 'not applied' in caplog.text
