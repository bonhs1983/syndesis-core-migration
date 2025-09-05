import json
import os
import logging
from datetime import datetime
from app import db
from models import InteractionLog, PipelineStatus
from config import Config

class InteractionLogger:
    """
    Logger for FractalAI agent interactions
    """
    
    def __init__(self):
        self.log_file = Config.INTERACTIONS_LOG_FILE
        self._ensure_log_file_exists()
        self._update_status('idle', 'Logger initialized')
        
    def _ensure_log_file_exists(self):
        """Ensure the log file and directory exist"""
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                pass  # Create empty file
                
    def _update_status(self, status, message=None):
        """Update the logger status in database"""
        try:
            logger_status = PipelineStatus.query.filter_by(component='logger').first()
            if not logger_status:
                logger_status = PipelineStatus(component='logger')
                db.session.add(logger_status)
            
            logger_status.status = status
            logger_status.last_activity = datetime.utcnow()
            if message:
                logger_status.message = message
            
            db.session.commit()
        except Exception as e:
            logging.error(f"Failed to update logger status: {e}")
    
    def log_interaction(self, agent_input, agent_output, context=None, session_id=None):
        """
        Log an agent interaction both to JSONL file and database
        """
        try:
            self._update_status('running', 'Logging interaction')
            
            interaction_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'agent_input': agent_input,
                'agent_output': agent_output,
                'context': context,
                'session_id': session_id
            }
            
            # Write to JSONL file
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(interaction_data, ensure_ascii=False) + '\n')
            
            # Save to database
            interaction = InteractionLog(
                agent_input=agent_input,
                agent_output=agent_output,
                context=context,
                session_id=session_id
            )
            db.session.add(interaction)
            db.session.commit()
            
            self._update_status('idle', f'Logged interaction for session {session_id}')
            logging.info(f"Interaction logged successfully for session {session_id}")
            
        except Exception as e:
            error_msg = f"Failed to log interaction: {e}"
            self._update_status('error', error_msg)
            logging.error(error_msg)
            raise
    
    def get_log_stats(self):
        """Get statistics about logged interactions"""
        try:
            # Count lines in JSONL file
            with open(self.log_file, 'r', encoding='utf-8') as f:
                jsonl_count = sum(1 for line in f if line.strip())
            
            # Count database records
            db_count = InteractionLog.query.count()
            
            return {
                'jsonl_count': jsonl_count,
                'db_count': db_count,
                'log_file_size': os.path.getsize(self.log_file) if os.path.exists(self.log_file) else 0
            }
        except Exception as e:
            logging.error(f"Failed to get log stats: {e}")
            return {'jsonl_count': 0, 'db_count': 0, 'log_file_size': 0}
