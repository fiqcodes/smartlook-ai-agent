import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
from langchain_core.messages import AIMessage

import agent


@pytest.mark.parametrize('sql', [
    'DELETE FROM users', 'SELECT 1; SELECT 2',
    'SELECT * FROM other-project.thelook_ecommerce.users',
    'SELECT * FROM bigquery-public-data.other.users',
    'SELECT * FROM secrets',
    "SELECT * FROM EXTERNAL_QUERY('connection', 'select 1')",
    'SELECT project.dataset.remote_fn(email) FROM users',
])
def test_rejects_unsafe_sql(sql):
    with pytest.raises(Exception):
        agent._validated_sql(sql)


def test_ctes_aliases_nested_limits_and_literals():
    sql = "WITH x AS (SELECT id FROM users LIMIT 1) SELECT * FROM x"
    result = agent._validated_sql(sql)
    assert result.endswith('LIMIT 100')
    assert 'bigquery-public-data' in result
    assert agent._validated_sql('SELECT * FROM users LIMIT 999999').endswith('LIMIT 1000')
    assert "'users'" in agent._validated_sql("SELECT 'users' AS label FROM users LIMIT 2")


def test_query_cost_timeout_and_row_bound(monkeypatch):
    job = Mock()
    rows = Mock()
    rows.schema = [SimpleNamespace(name='id')]
    rows.__iter__ = Mock(return_value=iter([{'id': 1}]))
    job.result.return_value = rows
    client = Mock()
    client.query.return_value = job
    monkeypatch.setattr(agent, 'get_bigquery_client', lambda: client)
    assert agent._execute_sql('SELECT id FROM users').iloc[0]['id'] == 1
    kwargs = client.query.call_args.kwargs
    assert kwargs['job_config'].maximum_bytes_billed == 1_000_000_000
    assert int(kwargs['job_config'].job_timeout_ms) == 45000
    assert job.result.call_args.kwargs['max_results'] == 1000


def test_timeout_attempts_cancellation(monkeypatch):
    job = Mock()
    job.result.side_effect = TimeoutError()
    client = Mock()
    client.query.return_value = job
    monkeypatch.setattr(agent, 'get_bigquery_client', lambda: client)
    with pytest.raises(TimeoutError):
        agent._execute_sql('SELECT id FROM users')
    job.cancel.assert_called_once()


def test_real_graph_routes_and_context_isolation(monkeypatch):
    class LLM:
        def invoke(self, messages):
            if isinstance(messages, list):
                prompt = messages[0].content
                question = prompt.split('Current User Question: ')[-1].split('\n')[0]
                return AIMessage(content=json.dumps({'tool': 'chat_with_user', 'args': {'message': question}}))
            return AIMessage(content=agent.get_conversation_context())
    monkeypatch.setattr(agent, 'make_llm', lambda **_: LLM())
    def run(name):
        return agent.ask(name, [{'role': 'User', 'content': name + '-private'}])
    with ThreadPoolExecutor(max_workers=2) as pool:
        a, b = list(pool.map(run, ['alice', 'bob']))
    assert a['tool_call']['args']['message'] == 'alice'
    assert b['tool_call']['args']['message'] == 'bob'
    assert 'alice-private' in a['response'] and 'bob-private' not in a['response']
    assert 'bob-private' in b['response'] and 'alice-private' not in b['response']
    assert agent._conversation_history.get() is None


def test_chart_queries_once_and_preserves_download_data(monkeypatch):
    responses = iter(['SELECT id FROM users', 'Insight from the existing rows'])
    llm = Mock()
    llm.invoke.side_effect = lambda *_: AIMessage(content=next(responses))
    monkeypatch.setattr(agent, 'make_llm', lambda **_: llm)
    execute = Mock(return_value=pd.DataFrame({'id': [1, 2]}))
    monkeypatch.setattr(agent, '_execute_sql', execute)
    monkeypatch.setattr(agent, '_create_visualization', lambda *_: {'chart_json': '{}'})
    result = json.loads(agent.create_visualization.invoke({'question': 'show users'}))
    assert result['response_text'] == 'Insight from the existing rows'
    assert len(result['source_data']) == 2
    execute.assert_called_once()


def test_product_count_chart_labels_match_live_result():
    frame = pd.DataFrame({'department': ['Women', 'Men'], 'product_count': [15989, 13131]})
    chart = json.loads(agent._create_visualization(frame, 'Product count by department', 'bar')['chart_json'])
    assert chart['layout']['xaxis']['title']['text'] == 'Department'
    assert chart['layout']['yaxis']['title']['text'] == 'Product Count'
    assert '$' not in chart['layout']['yaxis']['tickformat']
    assert '$' not in chart['data'][0]['texttemplate']


def test_small_counts_are_not_percentages_and_revenue_remains_money():
    frame = pd.DataFrame({'department': ['Women'], 'product_count': [1], 'revenue': [10000]})
    assert agent._detect_y_axis_label('product_count', 'Product count', frame) == ('Product Count', False, False, True)
    assert agent._detect_y_axis_label('revenue', 'Revenue by department', frame) == ('Revenue ($)', True, False, False)
