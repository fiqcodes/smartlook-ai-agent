"""SmartLook Flask entrypoint for local development and Vercel."""
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
from datetime import timedelta
from urllib.parse import urlsplit

from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request, session, send_from_directory
from werkzeug.exceptions import HTTPException

load_dotenv('.env.local')

from storage import configured_store, ConversationBusy, StorageUnavailable

logger = logging.getLogger(__name__)
REQUIRED_AI_ENV = ('GROQ_API_KEY', 'GOOGLE_CLOUD_PROJECT')

def ai_configured():
    return all(os.getenv(key) for key in REQUIRED_AI_ENV) and bool(
        os.getenv('GCP_SERVICE_ACCOUNT_JSON') or os.getenv('GCP_SERVICE_ACCOUNT')
    )
CID_PATTERN = re.compile(r'^[a-zA-Z0-9_-]{1,64}$')
MAX_RESPONSE_BYTES = 3_500_000


def create_app(test_config=None):
    app = Flask(__name__, static_folder=None)
    production = bool(os.getenv('VERCEL')) or os.getenv('APP_ENV') == 'production'
    app.config.update(
        PRODUCTION=production,
        SECRET_KEY=os.getenv('APP_SECRET_KEY') or (None if production else secrets.token_hex(32)),
        SESSION_COOKIE_NAME='smartlook_session',
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SECURE=production,
        SESSION_COOKIE_SAMESITE='Strict',
        PERMANENT_SESSION_LIFETIME=timedelta(seconds=int(os.getenv('SESSION_TTL_SECONDS', '86400'))),
        MAX_CONTENT_LENGTH=16_384,
        STORE=configured_store(),
        CHAT_LIMIT=int(os.getenv('CHAT_REQUESTS_PER_MINUTE', '5')),
        IP_LIMIT=int(os.getenv('IP_REQUESTS_PER_MINUTE', '10')),
        DAILY_LIMIT=int(os.getenv('DAILY_CHAT_LIMIT', '200')),
    )
    if test_config:
        app.config.update(test_config)

    def configuration_ready():
        return (bool(app.secret_key) and app.config['STORE'] is not None
                and (not app.config['PRODUCTION'] or (
                    len(app.secret_key) >= 32)))

    def ip_key():
        # Vercel overwrites x-vercel-forwarded-for; do not trust arbitrary x-forwarded-for.
        address = (request.headers.get('x-vercel-forwarded-for') if os.getenv('VERCEL') else None)
        address = (address or request.remote_addr or 'unknown').split(',')[0].strip()
        return hmac.new(app.secret_key.encode(), address.encode(), hashlib.sha256).hexdigest()

    def error(message, status):
        response = jsonify(error=message, status='error')
        response.status_code = status
        if status == 429:
            response.headers['Retry-After'] = '60'
        return response

    def cid():
        value = (request.get_json(silent=True) or {}).get('conversation_id') if request.method == 'POST' else request.args.get('conversation_id')
        if not isinstance(value, str) or not CID_PATTERN.fullmatch(value):
            raise ValueError('A valid conversation_id is required')
        return value

    @app.before_request
    def protect_requests():
        if request.path == '/api/health' or request.path.startswith('/assets/'):
            return None
        if not configuration_ready():
            if request.path.startswith('/api/'):
                return error('Service setup is incomplete. Please contact the owner.', 503)
            return render_template('unavailable.html'), 503
        if request.method in ('POST', 'PUT', 'PATCH', 'DELETE'):
            origin = request.headers.get('Origin')
            if request.headers.get('Sec-Fetch-Site') == 'cross-site' or (
                origin and urlsplit(origin).netloc != request.host
            ):
                return error('Cross-site requests are not allowed', 403)
        if 'sid' not in session:
            session['sid'] = secrets.token_urlsafe(32)
            session.permanent = True

    @app.after_request
    def response_headers(response):
        if not request.path.startswith('/assets/'):
            response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['Referrer-Policy'] = 'same-origin'
        return response

    @app.route('/login', methods=['GET', 'POST'])
    def login():
        # Keep old bookmarks working after opening the portfolio demo to everyone.
        return redirect('/')

    @app.route('/')
    def index():
        return send_from_directory(app.template_folder, 'index.html')

    @app.route('/assets/<path:filename>')
    def assets(filename):
        # Local fallback; Vercel serves public/assets directly from the CDN.
        return send_from_directory(os.path.join(app.root_path, 'public', 'assets'), filename)

    @app.post('/api/chat')
    def chat():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return error('A JSON object is required', 400)
        message = data.get('message')
        if not isinstance(message, str) or not message.strip() or len(message) > 4000:
            return error('Enter a message between 1 and 4000 characters', 400)
        conversation_id = cid()
        runner = app.config.get('AGENT_RUNNER')
        if runner is None:
            if not ai_configured():
                return error('AI credentials have not been configured by the owner', 503)
            from agent import ask
            runner = ask
        store = app.config['STORE']
        sid = session['sid']
        with store.lock(sid):
            limits = [('session:' + sid, app.config['CHAT_LIMIT'], 60),
                      ('ip:' + ip_key(), app.config['IP_LIMIT'], 60),
                      ('daily', app.config['DAILY_LIMIT'], 86400)]
            for key, limit, seconds in limits:
                if not store.allow(key, limit, seconds):
                    return error('Usage limit reached. Please try again later.', 429)
            history = store.load(sid, conversation_id)
            result = runner(message.strip(), history)
            raw = result['response']
            tool = (result.get('tool_call') or {}).get('name')
            try:
                structured = json.loads(raw)
            except (ValueError, TypeError):
                structured = None
            if not isinstance(structured, dict):
                structured = None
            is_visualization = bool(structured and tool in ('create_visualization', 'create_cohort_analysis'))
            is_data = bool(structured and structured.get('is_data_response'))
            # Send large chart/source data only once, not both as a JSON string and object.
            text = (structured.get('response_text') or structured.get('error') or 'Chart generated.') if structured else str(raw)
            payload = dict(
                response=text, status='success', tool_used=tool,
                context_aware=bool(history), is_visualization=is_visualization,
                visualization_data=structured if is_visualization else None,
                is_data_response=is_data, data_response=structured if is_data and not is_visualization else None,
            )
            response = jsonify(payload)
            if len(response.get_data()) > MAX_RESPONSE_BYTES:
                return error('Result is too large. Ask for fewer columns or a smaller date range.', 413)
            history += [{'role': 'User', 'content': message.strip()},
                        {'role': 'Assistant', 'content': text[:4000]}]
            store.save(sid, conversation_id, history[-20:])
            return response

    @app.post('/api/clear')
    def clear():
        store = app.config['STORE']
        with store.lock(session['sid']):
            store.clear(session['sid'])
        return jsonify(status='success')

    @app.get('/api/history')
    def history():
        messages = app.config['STORE'].load(session['sid'], cid())
        return jsonify(history=messages, count=len(messages), status='success')

    @app.get('/api/health')
    def health():
        # Liveness/readiness only: never spend Groq tokens or submit BigQuery jobs.
        ready = configuration_ready() and ai_configured()
        return jsonify(status='ready' if ready else 'setup_required'), 200 if ready else 503

    @app.errorhandler(ConversationBusy)
    def busy(exc):
        return error(str(exc), 409)

    @app.errorhandler(StorageUnavailable)
    def storage_error(exc):
        logger.warning('Shared storage request failed')
        return error('Conversation storage is unavailable. Please try again later.', 503)

    @app.errorhandler(ValueError)
    def invalid_input(exc):
        return error('Invalid request. Check the conversation ID and message.', 400)

    @app.errorhandler(HTTPException)
    def http_error(exc):
        return error(exc.name, exc.code)

    @app.errorhandler(Exception)
    def internal_error(exc):
        logger.error('Chat request failed (%s)', type(exc).__name__)
        return error('Unable to complete the request. Please try again.', 500)

    return app


app = create_app()

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=int(os.getenv('PORT', '8000')), debug=False)
