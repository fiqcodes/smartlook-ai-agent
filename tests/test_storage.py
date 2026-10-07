import json
from concurrent.futures import ThreadPoolExecutor

import fakeredis
import pytest
import requests

from storage import RedisStore, StorageUnavailable, ConversationBusy, configured_store


@pytest.fixture
def store(monkeypatch):
    redis = fakeredis.FakeRedis(decode_responses=True)
    class Reply:
        def __init__(self, command):
            self.command = command
        def raise_for_status(self):
            pass
        def json(self):
            return {'result': redis.execute_command(*self.command)}
    monkeypatch.setattr(requests, 'post', lambda url, **kw: Reply(kw['json']))
    return RedisStore('https://redis.example', 'test-token'), redis


def test_redis_history_namespace_ttl_and_clear(store):
    backend, redis = store
    backend.save('alice', 'one', [{'content': 'private'}])
    assert backend.load('alice', 'one') == [{'content': 'private'}]
    assert backend.load('bob', 'one') == []
    assert backend.load('alice', 'two') == []
    assert redis.ttl('smartlook:production:history:alice') > 86000
    backend.clear('alice')
    assert backend.load('alice', 'one') == []


def test_atomic_rate_limit(store):
    backend, _ = store
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: backend.allow('same', 5, 60), range(20)))
    assert sum(results) == 5


def test_lock_ownership_release_and_conversation_cap(store):
    backend, redis = store
    with backend.lock('alice'):
        with pytest.raises(ConversationBusy):
            with backend.lock('alice'):
                pass
        with backend.lock('bob'):
            pass
    with backend.lock('alice'):
        pass
    for i in range(30):
        backend.save('alice', str(i), [])
    with pytest.raises(StorageUnavailable):
        backend.save('alice', 'overflow', [])
    backend.save('alice', '0', [{'content': 'updated'}])


def test_transport_errors_fail_closed(monkeypatch):
    def fail(*_, **__):
        raise requests.Timeout('private-token')
    monkeypatch.setattr(requests, 'post', fail)
    with pytest.raises(StorageUnavailable, match='unavailable') as exc:
        RedisStore('https://redis.example', 'token').allow('x', 1, 60)
    assert 'private-token' not in str(exc.value)


def test_production_never_uses_memory_fallback(monkeypatch):
    for key in ('UPSTASH_REDIS_REST_URL', 'UPSTASH_REDIS_REST_TOKEN', 'KV_REST_API_URL', 'KV_REST_API_TOKEN'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('VERCEL', '1')
    assert configured_store() is None
