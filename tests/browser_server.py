"""Local-only browser fixture. Uses the real Flask app with a deterministic agent."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import create_app
from storage import MemoryStore


def fixture_agent(message, history):
    if 'chart' in message.lower():
        return {'tool_call': {'name': 'create_visualization'}, 'response': json.dumps({
            'response_text': 'Chart fixture: revenue by month.',
            'chart_json': json.dumps({'data': [{'type': 'bar', 'x': ['Jan', 'Feb'], 'y': [10, 20]}], 'layout': {'title': 'Revenue'}}),
            'source_data': [{'month': 'Jan', 'revenue': 10}, {'month': 'Feb', 'revenue': 20}],
            'sql': 'SELECT month, revenue FROM orders', 'question': message, 'row_count': 2,
        })}
    return {'response': f'Echo: {message}. Previous messages: {len(history)}', 'tool_call': {'name': 'chat_with_user'}}


if __name__ == '__main__':
    app = create_app(dict(PRODUCTION=False, SECRET_KEY='browser-fixture-only', ACCESS_PASSWORD='browser-test-password',
        STORE=MemoryStore(), CHAT_LIMIT=100, IP_LIMIT=100, DAILY_LIMIT=100, AGENT_RUNNER=fixture_agent))
    app.run(host='127.0.0.1', port=8765, debug=False)
