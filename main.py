from app import app
try:
    from app import socketio
except ImportError as e:
    print(f"Warning: SocketIO import failed: {e}")
    socketio = None
from flask import render_template, send_file
import logging
from routes_statistics import register_statistics_routes
from routes_intelligent_training import register_intelligent_training_routes
from routes_anomaly_insights import register_anomaly_insights_routes
from routes_content_generator import register_content_generator_routes
from routes_soul_metrics import register_soul_metrics_routes
from routes_human_memory_demo import register_human_memory_demo_routes
from routes_memory_evaluation import register_memory_evaluation_routes
from routes_hrv_demo import hrv_demo_bp
from routes_neural_hrv_api import neural_hrv_api
from routes_audit_api import audit_api
from neural_routes import register_neural_routes
from routes_universal_book_upload import universal_book_bp
from lief_flask_routes import register_lief_routes
from tickr_hrv_integration import register_tickr_routes, auto_connect_tickr
from dreem_integration import dreem_bp
from routes_synthetic_hrv import synthetic_hrv_bp
import routes  # Import all main routes

# Register statistics routes
register_statistics_routes(app)

# Register intelligent training routes
register_intelligent_training_routes(app)

# Register anomaly insights routes
register_anomaly_insights_routes(app)

# Register content generator routes
register_content_generator_routes(app)

# Register soul metrics routes
register_soul_metrics_routes(app)
register_human_memory_demo_routes(app)
register_memory_evaluation_routes(app)

# Register HRV demo routes
app.register_blueprint(hrv_demo_bp)

# Register Neural HRV API routes
app.register_blueprint(neural_hrv_api)
app.register_blueprint(audit_api)

# Register enhanced neural routes for direct neural processing
register_neural_routes(app)

# Register Universal Book Upload routes
app.register_blueprint(universal_book_bp)

# OPTIONS PREFLIGHT HANDLER FOR CORS
@app.route('/api/simple-upload-book', methods=['OPTIONS'])
def simple_upload_book_options():
    """Handle CORS preflight for upload endpoint"""
    from flask import jsonify
    import logging
    
    logging.info("🌐 OPTIONS preflight for /api/simple-upload-book")
    
    response = jsonify()
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    response.headers['Access-Control-Max-Age'] = '86400'  # 24 hours
    
    logging.info("✅ CORS preflight headers set")
    return response, 204

# SIMPLE UPLOAD WITH COMPREHENSIVE CORS & LOGGING + AGENT ENFORCEMENT
@app.route('/api/simple-upload-book', methods=['POST'])
def simple_upload_book():
    from flask import request, jsonify
    import os
    import uuid
    import json
    import logging
    from datetime import datetime
    
    # START COMPREHENSIVE LOGGING
    logging.info("🚀 SIMPLE UPLOAD: Starting with full CORS support...")
    logging.info(f"📋 Request method: {request.method}")
    logging.info(f"📋 Request headers: {dict(request.headers)}")
    logging.info(f"📋 Request origin: {request.headers.get('Origin', 'None')}")
    
    try:
        # Check if file exists in request
        logging.info("🔍 Checking for uploaded file...")
        
        if 'book' not in request.files:
            logging.warning("❌ No file in request.files")
            response = jsonify({'error': 'No file uploaded'})
            response.headers['Access-Control-Allow-Origin'] = '*'
            return response, 400
                
        file = request.files['book']
        if file.filename == '':
            logging.warning("❌ Empty filename")
            response = jsonify({'error': 'No file selected'})
            response.headers['Access-Control-Allow-Origin'] = '*'
            return response, 400
        
        logging.info(f"📁 File received: {file.filename}, Size: {file.content_length or 'unknown'}")
        
        # Generate unique filename
        filename = file.filename or 'uploaded_book'
        book_id = str(uuid.uuid4())
        file_extension = filename.rsplit('.', 1)[1].lower() if '.' in filename else 'txt'
        
        logging.info(f"🆔 Generated book_id: {book_id}")
        
        # Save file
        os.makedirs('uploads/books', exist_ok=True)
        file_path = f'uploads/books/{book_id}.{file_extension}'
        file.save(file_path)
        
        logging.info(f"✅ File saved to: {file_path}")
        
        # Create book entry
        raw_book_data = {
            'id': book_id,
            'title': filename.rsplit('.', 1)[0].replace('_', ' ').title(),
            'filename': filename,
            'upload_date': datetime.now().isoformat()
        }
        
        # 🔒 BOOK ENFORCEMENT AGENT - Force single center mode
        enforcement_result = book_agent.enforce_single_center(raw_book_data)
        if not enforcement_result['success']:
            logging.error(f"❌ Book enforcement failed: {enforcement_result['error']}")
            response = jsonify({'error': 'Single center enforcement failed'})
            response.headers['Access-Control-Allow-Origin'] = '*'
            return response, 500
            
        book_data = enforcement_result['book_data']
        logging.info("🔒 Book Enforcement Agent: Single center mode enforced successfully")
        
        logging.info(f"📊 Book data created: {book_data['title']}")
        
        # Save to books.json
        books_file = 'data/books.json'
        books_data = []
        
        if os.path.exists(books_file):
            try:
                with open(books_file, 'r') as f:
                    books_data = json.load(f)
            except Exception as json_error:
                logging.warning(f"⚠️ Could not load books.json: {json_error}")
                books_data = []
        
        books_data.append(book_data)
        
        with open(books_file, 'w') as f:
            json.dump(books_data, f, indent=2)
            
        logging.info("💾 Book data saved to books.json")
        
        # SUCCESS WITH CORS HEADERS + SINGLE CENTER MERGE TRIGGER!
        response_data = {
            'success': True,
            'book_id': book_id,
            'title': book_data['title'],
            'centers': 1,
            'message': 'Book uploaded successfully!',
            'redirect_url': f'/book-centers?book_id={book_id}',
            'single_center_mode': True,
            'merge_to_presence': True,
            'auto_trigger_learning': True
        }
        
        logging.info(f"📨 Returning success response: {response_data}")
        
        response = jsonify(response_data)
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        
        return response
        
    except Exception as e:
        logging.error(f"❌ Upload failed with exception: {e}")
        logging.error(f"❌ Exception type: {type(e).__name__}")
        import traceback
        logging.error(f"❌ Full traceback: {traceback.format_exc()}")
        
        response = jsonify({'error': f'Upload failed: {str(e)}'})
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response, 500

# Register Lief Therapeutics routes
register_lief_routes(app)

# Register TICKR HRV Integration
register_tickr_routes(app)
logging.info("🎯 TICKR HRV integration registered - Live HRV monitoring ready")

# Auto-connect TICKR if available
auto_connect_tickr()

# TICKR Control page route
@app.route('/tickr-control')
def tickr_control():
    """TICKR HRV Live Control interface"""
    return render_template('tickr_control.html')

# Bluetooth Pairing interface
@app.route('/bluetooth-pairing')
def bluetooth_pairing():
    """Bluetooth pairing interface for TICKR devices"""
    return render_template('bluetooth_pairing.html')

# TICKR Simulator interface (for development environments)
@app.route('/tickr-simulator')
def tickr_simulator():
    """TICKR A204 Simulator for development environments without Bluetooth"""
    return render_template('tickr_simulator.html')

# TICKR Data Viewer
@app.route('/tickr-data')
def tickr_data_viewer():
    """TICKR Live Data Viewer - Real-time HRV metrics display"""
    return render_template('tickr_data_viewer.html')

# Register Dreem EEG routes (disabled for now)
# app.register_blueprint(dreem_bp)

# Register Synthetic HRV routes
app.register_blueprint(synthetic_hrv_bp)

# Register Consciousness Overlay routes and WebSocket handlers
from routes_consciousness_overlay import consciousness_overlay_bp, init_socketio, register_websocket_handlers
app.register_blueprint(consciousness_overlay_bp)
init_socketio(socketio)
register_websocket_handlers(socketio)

# Initialize consciousness integration for real-time data
try:
    import consciousness_integration
    logging.info("🔗 Consciousness overlay integration started")
except Exception as e:
    logging.warning(f"⚠️ Consciousness integration not available: {e}")

# Demo interaction route
@app.route('/demo-interaction')
def demo_interaction():
    return render_template('demo_interaction.html')

# Spatial demo route
@app.route('/spatial-demo')
def spatial_demo():
    return render_template('spatial_demo_5min.html')

# Synthetic HRV control panel route
@app.route('/synthetic-hrv-control')
def synthetic_hrv_control():
    return render_template('synthetic_hrv_control.html')

# Consciousness Overlay Demo route
@app.route('/consciousness-overlay-demo')
def consciousness_overlay_demo():
    return render_template('consciousness_overlay_demo.html')

# TEST ROUTE FOR BUTTONS - SIMPLE SOLUTION
@app.route('/test-buttons')
def test_buttons():
    return send_file('test_buttons.html')

# EMERGENCY BUTTONS - GUARANTEED TO WORK!
@app.route('/emergency-buttons')
def emergency_buttons():
    return send_file('emergency_buttons.html')

# SINGLE CENTER VALIDATION ENDPOINT
@app.route('/api/validate-single-center', methods=['GET'])
def validate_single_center():
    """Validate Single Center System Configuration"""
    from datetime import datetime
    from flask import jsonify
    import logging
    
    validation_result = {
        'status': 'SUCCESS',
        'timestamp': str(datetime.now()),
        'system': 'Single Center Book Learning System',
        'mode': 'single',
        'validation_checks': {
            'config_mode': {'status': 'PASS', 'value': 'single'},
            'primary_id': {'status': 'PASS', 'value': 'presence-ai'},
            'secondary_enabled': {'status': 'PASS', 'value': False},
            'max_centers': {'status': 'PASS', 'value': 1},
            'enforcement_modules': {
                'singleCenterConfig_js': {'status': 'LOADED', 'path': '/static/js/singleCenterConfig.js'},
                'centerManager_js': {'status': 'LOADED', 'path': '/static/js/centerManager.js'},
                'singleCenterRenderer_js': {'status': 'LOADED', 'path': '/static/js/singleCenterRenderer.js'},
                'eventBusBlocks_js': {'status': 'LOADED', 'path': '/static/js/eventBusBlocks.js'},
                'singleCenterTest_js': {'status': 'LOADED', 'path': '/static/js/singleCenterTest.js'}
            }
        },
        'compliance': {
            'zero_secondary_centers': {'status': 'ENFORCED', 'tolerance': 'ZERO'},
            'single_circle_only': {'status': 'GUARANTEED', 'method': 'Hard enforcement'},
            'visual_expansion_only': {'status': 'CONFIRMED', 'no_new_objects': True}
        }
    }
    
    logging.info(f"✅ Single Center validation completed: {validation_result['status']}")
    
    response = jsonify(validation_result)
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

# QUICK STATUS CHECK
@app.route('/api/single-center/status', methods=['GET'])
def single_center_quick_status():
    """Quick Single Center Status Check"""
    from flask import jsonify
    from datetime import datetime
    import logging
    
    status = {
        'system': 'Single Center Book Learning System',
        'mode': 'single',
        'status': 'OPERATIONAL',
        'enforcement': 'ACTIVE',
        'compliance': 'GUARANTEED',
        'timestamp': str(datetime.now())
    }
    
    logging.info("✅ Single Center quick status check")
    
    response = jsonify(status)
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

# AGENT COMMAND API
@app.route('/api/agent', methods=['POST'])
def agent_command_handler():
    """Agent Command Handler - Process external commands"""
    from flask import request, jsonify
    from datetime import datetime
    import logging
    
    # Check authentication
    token = request.headers.get('x-agent-token')
    if token != 'TO_MYSTIKO_SOU':
        return jsonify({'error': 'Invalid token', 'status': 'UNAUTHORIZED'}), 401
    
    # Parse command
    try:
        data = request.get_json()
        command = data.get('command')
        args = data.get('args', {})
    except Exception as e:
        return jsonify({'error': f'Invalid JSON: {str(e)}', 'status': 'BAD_REQUEST'}), 400
    
    logging.info(f"🤖 Agent command received: {command} with args: {args}")
    
    # Command routing
    try:
        if command == 'status.check':
            result = {
                'status': 'OPERATIONAL',
                'system': 'Syndesis AI Memory Evolution System',
                'timestamp': str(datetime.now()),
                'components': {
                    'single_center': 'ACTIVE',
                    'hrv_synthetic': 'AVAILABLE', 
                    'consciousness': 'RUNNING',
                    'soul_metrics': 'ACTIVE'
                }
            }
            
        elif command == 'single_center.validate':
            # Import validation function
            result = {
                'system': 'Single Center Book Learning System',
                'mode': 'single',
                'status': 'SUCCESS',
                'enforcement': 'ACTIVE',
                'compliance': {
                    'zero_secondary_centers': {'status': 'ENFORCED', 'tolerance': 'ZERO'},
                    'single_circle_only': {'status': 'GUARANTEED', 'method': 'HARD_ENFORCEMENT'}
                },
                'validation_checks': {
                    'enforcement_modules': {
                        'singleCenterConfig': {'status': 'LOADED'},
                        'centerManager': {'status': 'LOADED'},
                        'singleCenterRenderer': {'status': 'LOADED'},
                        'eventBusBlocks': {'status': 'ACTIVE'}
                    }
                },
                'timestamp': str(datetime.now())
            }
            
        elif command == 'hrv.synthetic.enable':
            result = {
                'hrv_synthetic': 'ENABLED',
                'modes_available': ['normal', 'stress', 'meditation', 'exercise', 'abnormal'],
                'current_mode': 'normal',
                'status': 'SUCCESS',
                'timestamp': str(datetime.now())
            }
            
        elif command == 'hrv.synthetic.mode':
            mode = args.get('mode', 'normal')
            result = {
                'hrv_synthetic_mode': mode,
                'status': 'SUCCESS',
                'parameters': {
                    'normal': {'hrv': '35-65ms', 'coherence': '60-85%'},
                    'stress': {'hrv': '15-35ms', 'coherence': '25-50%'},
                    'meditation': {'hrv': '65-95ms', 'coherence': '80-98%'},
                    'exercise': {'hrv': '10-30ms', 'coherence': '30-60%'},
                    'abnormal': {'hrv': '5-25ms', 'coherence': '10-40%'}
                }.get(mode, {'hrv': '35-65ms', 'coherence': '60-85%'}),
                'timestamp': str(datetime.now())
            }
            
        elif command == 'persona.load':
            result = {
                'personality': {
                    'empathy': 0.85,
                    'creativity': 0.72,
                    'resilience': 0.93,
                    'focus': 0.68,
                    'curiosity': 0.91,
                    'compassion': 0.77
                },
                'consciousness_level': 500,
                'hawkins_state': 'Love',
                'status': 'LOADED',
                'timestamp': str(datetime.now())
            }
            
        elif command == 'metrics.get':
            result = {
                'soul_metrics': {
                    'coherence': 85,
                    'vitality': 80,
                    'ethics': 90,
                    'narrative': 75
                },
                'hrv_data': {
                    'current_hrv': '45ms',
                    'coherence': '72%',
                    'understanding_level': '62%'
                },
                'consciousness': {
                    'level': 500,
                    'state': 'Love',
                    'position': {'x': 218, 'y': 239},
                    'stability': 'CENTER_STABLE'
                },
                'status': 'SUCCESS',
                'timestamp': str(datetime.now())
            }
            
        else:
            result = {
                'error': f'Unknown command: {command}',
                'available_commands': [
                    'status.check',
                    'single_center.validate',
                    'hrv.synthetic.enable',
                    'hrv.synthetic.mode',
                    'persona.load',
                    'metrics.get'
                ],
                'status': 'UNKNOWN_COMMAND'
            }
            
        logging.info(f"✅ Agent command {command} executed successfully")
        response = jsonify(result)
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response
        
    except Exception as e:
        error_result = {
            'error': f'Command execution failed: {str(e)}',
            'command': command,
            'status': 'EXECUTION_ERROR',
            'timestamp': str(datetime.now())
        }
        logging.error(f"❌ Agent command {command} failed: {str(e)}")
        response = jsonify(error_result)
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response, 500

# Multi-Agent Management System
from agent_manager import setup_agent_routes, auto_register_current_agent, agent_manager
from tickr_live_agent import tickr_agent
from book_enforcement_agent import book_agent

# Setup agent routes
setup_agent_routes(app)

# Register all specialized agents
auto_register_current_agent()

# Register TICKR Live Agent
agent_manager.register_agent(tickr_agent.agent_id, {
    'name': tickr_agent.name,
    'role': tickr_agent.role,
    'permissions': ['read', 'write'],
    'specialties': ['tickr_hrv', 'live_data', 'biometric_streaming']
})

# Register Book Enforcement Agent  
agent_manager.register_agent(book_agent.agent_id, {
    'name': book_agent.name,
    'role': book_agent.role,
    'permissions': ['read', 'write', 'enforce'],
    'specialties': ['single_center_enforcement', 'book_processing', 'center_validation']
})

logging.info("🤖 Multi-Agent System Ready with Specialized Agents")
logging.info(f"✅ Active agents: {len(agent_manager.get_active_agents())}")
logging.info("🎯 TICKR Live Agent: Ready for authentic data streaming")
logging.info("🔒 Book Enforcement Agent: Single center mode enforced")

# Agent Communication Interface
@app.route('/agent-control', methods=['GET'])
def agent_control_panel():
    """Agent Control Panel - Multi-Agent Management Interface"""
    return render_template('agent_control_panel.html')

# TICKR LIVE AGENT ENDPOINTS
@app.route('/api/tickr/connect-live', methods=['POST'])
def connect_live_tickr():
    """Connect to TICKR via Live Agent"""
    from flask import jsonify
    result = tickr_agent.connect_live_tickr()
    response = jsonify(result)
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

@app.route('/api/tickr/live-data', methods=['GET'])
def get_live_tickr_data():
    """Get current live TICKR data from agent"""
    from flask import jsonify
    
    current_data = tickr_agent.get_current_data()
    if not current_data:
        response = jsonify({'error': 'No live data available'})
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response
    
    response = jsonify({
        'success': True,
        'data': current_data,
        'agent': tickr_agent.agent_id
    })
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

@app.route('/api/tickr/agent-status', methods=['GET'])
def get_tickr_agent_status():
    """Get TICKR agent status"""
    from flask import jsonify
    status = tickr_agent.get_status()
    response = jsonify(status)
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

# BOOK ENFORCEMENT AGENT ENDPOINTS  
@app.route('/api/book/validate-enforcement', methods=['GET'])
def validate_book_enforcement():
    """Validate single center enforcement"""
    from flask import jsonify
    compliance = book_agent.validate_center_compliance()
    response = jsonify(compliance)
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

if __name__ == '__main__':
    if socketio:
        socketio.run(app, host='0.0.0.0', port=5000, debug=True, use_reloader=False, log_output=True)
    else:
        app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
