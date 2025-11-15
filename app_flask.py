from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import logging
from agent_cloud import ask, check_connection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')
CORS(app)

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('templates', 'index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        logger.info(f"Received message: {message}")
        
        # Call your agent
        response = ask(message)
        
        logger.info(f"Agent response: {response[:100]}...")
        
        return jsonify({
            'response': response,
            'status': 'success'
        })
    
    except Exception as e:
        logger.error(f"Error processing chat: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Check system health"""
    try:
        status = check_connection()
        return jsonify({
            'status': 'healthy',
            'connections': status
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/assets/<path:filename>')
def serve_assets(filename):
    return send_from_directory('templates/assets', filename)

if __name__ == '__main__':
    # Create directories if they don't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    # Run the app on port 8000 (avoiding macOS AirPlay on 5000)
    port = int(os.environ.get('PORT', 8000))
    print(f"\n🚀 Starting TheLook Ecommerce AI Analyst on http://localhost:{port}")
    print(f"🛍️ Open your browser and visit: http://localhost:{port}\n")
    app.run(host='0.0.0.0', port=port, debug=True)