from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import logging
from agent_cloud import ask, check_connection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')

# Enable CORS for all routes
CORS(app)

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('templates', 'index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages with conversation memory"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        logger.info(f"Received message: {message}")
        
        # Call your agent (automatically uses conversation history)
        response = ask(message)
        
        # Get tool routing info for debugging
        tool_call = get_last_tool_call()
        
        logger.info(f"Tool used: {tool_call['name'] if tool_call else 'unknown'}")
        logger.info(f"Agent response: {response[:100]}...")
        
        return jsonify({
            'response': response,
            'status': 'success',
            'tool_used': tool_call['name'] if tool_call else None,
            'context_aware': len(get_conversation_history()) > 0
        })
    
    except Exception as e:
        logger.error(f"Error processing chat: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/clear', methods=['POST'])
def clear_conversation():
    """Clear conversation history"""
    try:
        clear_history()
        logger.info("Conversation history cleared")
        return jsonify({
            'status': 'success',
            'message': 'Conversation history cleared'
        })
    except Exception as e:
        logger.error(f"Error clearing history: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    """Get conversation history (for debugging)"""
    try:
        history = get_conversation_history()
        return jsonify({
            'history': history,
            'count': len(history),
            'status': 'success'
        })
    except Exception as e:
        logger.error(f"Error getting history: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Check system health"""
    try:
        status = check_connection()
        history_count = len(get_conversation_history())
        
        return jsonify({
            'status': 'healthy',
            'connections': status,
            'conversation_messages': history_count,
            'memory_enabled': True
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/assets/<path:filename>')
def serve_assets(filename):
    """Serve static assets"""
    return send_from_directory('templates/assets', filename)

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'status': 'error'
    }), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({
        'error': 'Internal server error',
        'status': 'error'
    }), 500

if __name__ == '__main__':
    # Create directories if they don't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates/assets', exist_ok=True)
    
    # Run the app on port 8000 (avoiding macOS AirPlay on 5000)
    port = int(os.environ.get('PORT', 8000))
    
    print("\n" + "="*80)
    print("🚀 SmartLook")
    print("="*80)
    print(f"📍 URL: http://localhost:{port}")
    print(f"💬 Memory: Enabled (tracks last 10 exchanges)")
    print(f"🧠 Context-aware follow-ups: Active")
    print("\n💡 API Endpoints:")
    print(f"   POST /api/chat         - Send messages")
    print(f"   POST /api/clear        - Clear conversation history")
    print(f"   GET  /api/history      - View conversation history")
    print(f"   GET  /api/health       - Check system health")
    print("="*80 + "\n")
    
    app.run(host='0.0.0.0', port=port, debug=True)