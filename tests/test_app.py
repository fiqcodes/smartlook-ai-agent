import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from app import create_app
from storage import MemoryStore, StorageUnavailable


@pytest.fixture
def app():
    return create_app(dict(TESTING=True, PRODUCTION=False, SECRET_KEY='test-secret',
        ACCESS_PASSWORD=None, STORE=MemoryStore(),
        CHAT_LIMIT=100, IP_LIMIT=100, DAILY_LIMIT=100,
        AGENT_RUNNER=lambda message, history: {
            'response': f'{message}; previous={len(history)}',
            'tool_call': {'name': 'chat_with_user'},
        }))


def chat(client, message='hello', cid='chat-one'):
    return client.post('/api/chat', json={'message': message, 'conversation_id': cid})


def test_separate_visitors_and_conversations(app):
    a, b = app.test_client(), app.test_client()
    assert chat(a).json['response'] == 'hello; previous=0'
    assert chat(a).json['response'] == 'hello; previous=2'
    assert chat(b).json['response'] == 'hello; previous=0'
    assert chat(a, cid='chat-two').json['response'] == 'hello; previous=0'
    assert chat(a).json['response'] == 'hello; previous=4'
    assert b.get('/api/history?conversation_id=chat-one').json['count'] == 2
    assert a.post('/api/clear').status_code == 200
    assert a.get('/api/history?conversation_id=chat-one').json['count'] == 0
    assert b.get('/api/history?conversation_id=chat-one').json['count'] == 2


def test_history_shared_between_app_instances_and_bounded(app):
    a = app.test_client()
    for _ in range(12):
        assert chat(a).status_code == 200
    other = create_app({**app.config, 'STORE': app.config['STORE']}).test_client()
    cookie = a.get_cookie('smartlook_session')
    other.set_cookie('smartlook_session', cookie.value)
    assert other.get('/api/history?conversation_id=chat-one').json['count'] == 20


def test_password_gate_and_cross_site_rejection(app):
    app.config['ACCESS_PASSWORD'] = 'long-test-password'
    c = app.test_client()
    assert c.get('/').status_code == 302
    assert chat(c).status_code == 401
    assert c.post('/login', data={'password': 'bad'}).status_code == 401
    assert c.post('/login', data={'password': 'long-test-password'}, headers={'Origin': 'https://evil.test'}).status_code == 403
    assert c.post('/login', data={'password': 'long-test-password'}).status_code == 302
    assert chat(c).status_code == 200
    assert c.post('/api/clear', headers={'Sec-Fetch-Site': 'cross-site'}).status_code == 403


def test_limits_fail_closed(app):
    app.config['CHAT_LIMIT'] = 1
    c = app.test_client()
    assert chat(c).status_code == 200
    assert chat(c).status_code == 429
    app.config['DAILY_LIMIT'] = 1
    assert chat(app.test_client()).status_code == 429
    app.config['STORE'] = None
    assert chat(c).status_code == 503


def test_invalid_payloads(app):
    c = app.test_client()
    for value in [None, [], 'hi', {'message': 123}, {'message': 'hi', 'conversation_id': '../other'}]:
        assert c.post('/api/chat', json=value).status_code == 400
    assert chat(c, 'x' * 4001).status_code == 400
    assert c.post('/api/chat', data='x' * 20000, content_type='application/json').status_code == 413


def test_no_duplicate_source_data_and_payload_limit(app):
    data = {'response_text': 'A chart', 'source_data': [{'x': 1}], 'chart_json': '{}'}
    app.config['AGENT_RUNNER'] = lambda *_: {'response': json.dumps(data), 'tool_call': {'name': 'create_visualization'}}
    response = chat(app.test_client()).json
    assert response['response'] == 'A chart'
    assert response['visualization_data'] == data
    assert response['data_response'] is None
    app.config['AGENT_RUNNER'] = lambda *_: {'response': 'a' * 3_500_001}
    assert chat(app.test_client()).status_code == 413


def test_no_secrets_or_raw_provider_errors(app):
    def broken(*_):
        raise RuntimeError('secret-provider-key')
    app.config['AGENT_RUNNER'] = broken
    response = chat(app.test_client())
    assert response.status_code == 500
    assert 'secret-provider-key' not in response.text


def test_same_session_concurrency_and_clear_lock(app):
    started, release = Event(), Event()
    def slow(*_):
        started.set()
        assert release.wait(5)
        return {'response': 'done'}
    app.config['AGENT_RUNNER'] = slow
    a, b = app.test_client(), app.test_client()
    a.get('/')
    b.set_cookie('smartlook_session', a.get_cookie('smartlook_session').value)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(chat, a)
        assert started.wait(3)
        try:
            assert chat(b).status_code == 409
            assert b.post('/api/clear').status_code == 409
        finally:
            release.set()
        assert future.result().status_code == 200


def test_health_has_no_external_calls(app, monkeypatch):
    for key in ('GROQ_API_KEY', 'GOOGLE_CLOUD_PROJECT', 'GCP_SERVICE_ACCOUNT_JSON'):
        monkeypatch.delenv(key, raising=False)
    response = app.test_client().get('/api/health')
    assert response.status_code == 503
    assert response.json == {'status': 'setup_required'}
    assert app.test_client().get('/').status_code == 200
    assert app.test_client().get('/assets/smartlook_logo.png').status_code == 200


def test_google_credential_alias_and_canonical_names(app, monkeypatch):
    monkeypatch.setenv('GROQ_API_KEY', 'test-only')
    monkeypatch.setenv('GOOGLE_CLOUD_PROJECT', 'test-project')
    monkeypatch.delenv('GCP_SERVICE_ACCOUNT_JSON', raising=False)
    monkeypatch.setenv('GCP_SERVICE_ACCOUNT', '{}')
    assert app.test_client().get('/api/health').json == {'status': 'ready'}
    monkeypatch.delenv('GCP_SERVICE_ACCOUNT', raising=False)
    assert app.test_client().get('/api/health').status_code == 503
    monkeypatch.setenv('GCP_SERVICE_ACCOUNT_JSON', '{}')
    assert app.test_client().get('/api/health').json == {'status': 'ready'}
