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
    try:
        return send_from_directory('templates', 'index.html')
    except Exception as e:
        logger.error(f"Error serving index: {str(e)}")
        return jsonify({'error': 'Failed to load page'}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        logger.info(f"📨 Received: {message[:50]}...")
        
        # Call your agent
        response = ask(message)
        
        logger.info(f"✅ Response sent: {len(response)} chars")
        
        return jsonify({
            'response': response,
            'status': 'success'
        })
    
    except Exception as e:
        logger.error(f"❌ Error in chat: {str(e)}")
        return jsonify({
            'error': f'Internal server error: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint - used by Render/Railway"""
    try:
        status = check_connection()
        
        if status['groq'] and status['bigquery']:
            return jsonify({
                'status': 'healthy',
                'connections': status,
                'message': '✅ All systems operational'
            }), 200
        else:
            return jsonify({
                'status': 'degraded',
                'connections': status,
                'message': '⚠️ Some services unavailable'
            }), 503
            
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 503

@app.route('/assets/<path:filename>')
def serve_assets(filename):
    """Serve static assets (images, etc)"""
    try:
        return send_from_directory('templates/assets', filename)
    except Exception as e:
        logger.error(f"Asset not found: {filename}")
        return jsonify({'error': 'Asset not found'}), 404

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    logger.error(f"Internal error: {str(e)}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Create directories if they don't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('templates/assets', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    # Get port from environment (Render/Railway will set this)
    port = int(os.environ.get('PORT', 8000))
    
    # Determine if running in production
    is_production = os.environ.get('RENDER') or os.environ.get('RAILWAY_ENVIRONMENT')
    
    print("\n" + "="*60)
    print("🚀 SmartLook AI Data Analyst")
    print("="*60)
    print(f"📍 Mode: {'PRODUCTION' if is_production else 'DEVELOPMENT'}")
    print(f"🌐 Port: {port}")
    print(f"🔗 URL: http://{'0.0.0.0' if is_production else 'localhost'}:{port}")
    print("="*60 + "\n")
    
    # Run the app
    # Use 0.0.0.0 for production (required for Render/Railway)
    # Use 127.0.0.1 for local development
    host = '0.0.0.0' if is_production else '127.0.0.1'
    debug = not is_production
    
    app.run(host=host, port=port, debug=debug)