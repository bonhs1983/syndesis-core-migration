import json
import os
import logging
from datetime import datetime
from app import db
from models import InteractionLog, PipelineStatus
from config import Config

# Try to import datasets, fallback if not available
try:
    from datasets import Dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    Dataset = None

class DatasetExporter:
    """
    Export logged interactions to HuggingFace Dataset format
    """
    
    def __init__(self):
        self.output_dir = Config.DATASETS_DIR
        self._update_status('idle', 'Exporter initialized')
        
    def _update_status(self, status, message=None):
        """Update the exporter status in database"""
        try:
            exporter_status = PipelineStatus.query.filter_by(component='exporter').first()
            if not exporter_status:
                exporter_status = PipelineStatus(component='exporter')
                db.session.add(exporter_status)
            
            exporter_status.status = status
            exporter_status.last_activity = datetime.utcnow()
            if message:
                exporter_status.message = message
            
            db.session.commit()
        except Exception as e:
            logging.error(f"Failed to update exporter status: {e}")
    
    def export_from_jsonl(self, jsonl_path=None):
        """
        Export interactions from JSONL file to Dataset format
        """
        if not jsonl_path:
            jsonl_path = Config.INTERACTIONS_LOG_FILE
            
        try:
            self._update_status('running', 'Reading JSONL file')
            
            # Read JSONL file
            interactions = []
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        interactions.append({
                            'input': data['agent_input'],
                            'output': data['agent_output'],
                            'timestamp': data['timestamp'],
                            'session_id': data.get('session_id', ''),
                            'context': data.get('context', '')
                        })
            
            if not interactions:
                raise ValueError("No interactions found in JSONL file")
                
            self._update_status('running', f'Creating dataset with {len(interactions)} interactions')
            
            # Create output directory
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            dataset_path = os.path.join(self.output_dir, f'interactions_dataset_{timestamp}')
            os.makedirs(dataset_path, exist_ok=True)
            
            # Save as JSON for easier inspection
            json_path = os.path.join(dataset_path, 'dataset.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(interactions, f, ensure_ascii=False, indent=2)
            
            # If HuggingFace datasets is available, create HF dataset too
            if HAS_DATASETS:
                dataset = Dataset.from_list(interactions)
                dataset.save_to_disk(dataset_path)
                logging.info("HuggingFace Dataset created successfully")
            else:
                logging.warning("HuggingFace datasets not available, exported as JSON only")
            
            # Mark interactions as processed
            self._mark_interactions_processed()
            
            self._update_status('idle', f'Dataset exported to {dataset_path}')
            logging.info(f"Dataset exported successfully to {dataset_path}")
            
            return dataset_path
            
        except Exception as e:
            error_msg = f"Failed to export dataset: {e}"
            self._update_status('error', error_msg)
            logging.error(error_msg)
            raise
    
    def export_from_database(self):
        """
        Export interactions from database to Dataset format
        """
        try:
            self._update_status('running', 'Reading from database')
            
            # Get unprocessed interactions
            interactions_query = InteractionLog.query.filter_by(processed=False).all()
            
            if not interactions_query:
                raise ValueError("No unprocessed interactions found in database")
            
            interactions = []
            for interaction in interactions_query:
                interactions.append({
                    'input': interaction.agent_input,
                    'output': interaction.agent_output,
                    'timestamp': interaction.timestamp.isoformat(),
                    'session_id': interaction.session_id or '',
                    'context': interaction.context or ''
                })
            
            self._update_status('running', f'Creating dataset with {len(interactions)} interactions')
            
            # Create output directory
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            dataset_path = os.path.join(self.output_dir, f'interactions_dataset_{timestamp}')
            os.makedirs(dataset_path, exist_ok=True)
            
            # Save as JSON for easier inspection
            json_path = os.path.join(dataset_path, 'dataset.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(interactions, f, ensure_ascii=False, indent=2)
            
            # If HuggingFace datasets is available, create HF dataset too
            if HAS_DATASETS:
                dataset = Dataset.from_list(interactions)
                dataset.save_to_disk(dataset_path)
                logging.info("HuggingFace Dataset created successfully")
            else:
                logging.warning("HuggingFace datasets not available, exported as JSON only")
            
            # Mark interactions as processed
            for interaction in interactions_query:
                interaction.processed = True
            db.session.commit()
            
            self._update_status('idle', f'Dataset exported to {dataset_path}')
            logging.info(f"Dataset exported successfully to {dataset_path}")
            
            return dataset_path
            
        except Exception as e:
            error_msg = f"Failed to export dataset from database: {e}"
            self._update_status('error', error_msg)
            logging.error(error_msg)
            raise
    
    def _mark_interactions_processed(self):
        """Mark all interactions as processed"""
        try:
            InteractionLog.query.update({InteractionLog.processed: True})
            db.session.commit()
            logging.info("Marked all interactions as processed")
        except Exception as e:
            logging.error(f"Failed to mark interactions as processed: {e}")
