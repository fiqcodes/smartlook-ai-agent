from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import logging
import json
from agent import ask, check_connection, clear_history, get_conversation_history, get_last_tool_call

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
    """Handle chat messages with conversation memory and visualization support"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        logger.info(f"Received message: {message}")
        
        # Call agent
        response = ask(message)
        
        # Get tool routing info
        tool_call = get_last_tool_call()
        
        logger.info(f"Tool used: {tool_call['name'] if tool_call else 'unknown'}")
        
        # Check if response is visualization data (JSON)
        is_visualization = False
        is_data_response = False
        visualization_data = None
        data_response = None
        
        # Check if tool is create_visualization OR create_cohort_analysis
        if tool_call and tool_call['name'] in ['create_visualization', 'create_cohort_analysis']:
            try:
                visualization_data = json.loads(response)
                is_visualization = True
                
                # Generate analysis text for visualizations (if not already present)
                if not visualization_data.get('error') and not visualization_data.get('response_text'):
                    # Import the tool directly
                    from agent import answer_ecommerce_question
                    
                    # Get the original question
                    question = visualization_data.get('question', message)
                    
                    # Call answer_ecommerce_question to get formatted analysis with recommendations
                    try:
                        analysis_response = answer_ecommerce_question.invoke({"question": question})
                        
                        # NEW: Parse the JSON response to extract just the text
                        try:
                            parsed_response = json.loads(analysis_response)
                            if parsed_response.get('is_data_response'):
                                visualization_data['response_text'] = parsed_response.get('response_text', '')
                            else:
                                visualization_data['response_text'] = analysis_response
                        except json.JSONDecodeError:
                            # If not JSON, use as-is (backward compatibility)
                            visualization_data['response_text'] = analysis_response
                            
                        logger.info("Generated analysis with actionable recommendations")
                    except Exception as e:
                        logger.warning(f"Could not generate analysis text: {e}")
                
                logger.info(f"Visualization created successfully: {tool_call['name']}")
            except json.JSONDecodeError:
                logger.warning("Failed to parse visualization JSON")
        
        # NEW: Check if tool is answer_ecommerce_question (data response with CSV)
        elif tool_call and tool_call['name'] in ['answer_ecommerce_question', 'generate_and_show_sql']:
            try:
                data_response = json.loads(response)
                if data_response.get('is_data_response'):
                    is_data_response = True
                    logger.info("Data response with CSV download created successfully")
                else:
                    # Not a data response, treat as regular text
                    data_response = None
            except json.JSONDecodeError:
                # Not JSON, treat as regular text response
                logger.info("Regular text response from answer_ecommerce_question")
        
        return jsonify({
            'response': response,
            'status': 'success',
            'tool_used': tool_call['name'] if tool_call else None,
            'context_aware': len(get_conversation_history()) > 0,
            'is_visualization': is_visualization,
            'visualization_data': visualization_data,
            'is_data_response': is_data_response,
            'data_response': data_response
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
    """Get conversation history"""
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
            'memory_enabled': True,
            'visualization_enabled': True,
            'cohort_analysis_enabled': True,
            'csv_download_enabled': True  # NEW
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
    return jsonify({
        'error': 'Endpoint not found',
        'status': 'error'
    }), 404

@app.errorhandler(500)
def internal_error(e):
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
    
    app.run(host='0.0.0.0', port=port, debug=True)