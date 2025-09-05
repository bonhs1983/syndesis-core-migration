import logging
from pipeline.dataset_exporter import DatasetExporter
from pipeline.trainer import LLMTrainer
from models import PipelineStatus

class PipelineOrchestrator:
    """
    Orchestrates the entire pipeline from logging to training
    """
    
    def __init__(self):
        self.exporter = DatasetExporter()
        self.trainer = LLMTrainer()
        
    def export_to_dataset(self, use_database=True):
        """
        Export interactions to HuggingFace Dataset
        """
        try:
            if use_database:
                dataset_path = self.exporter.export_from_database()
            else:
                dataset_path = self.exporter.export_from_jsonl()
            
            logging.info(f"Pipeline: Dataset exported to {dataset_path}")
            return dataset_path
            
        except Exception as e:
            logging.error(f"Pipeline: Failed to export dataset: {e}")
            raise
    
    def start_training(self, dataset_path=None, epochs=3):
        """
        Start LLM training with exported dataset
        """
        try:
            # If no dataset path provided, export first
            if not dataset_path:
                dataset_path = self.export_to_dataset()
            
            job_id = self.trainer.start_training(dataset_path, epochs=epochs)
            
            logging.info(f"Pipeline: Training job {job_id} started with dataset {dataset_path}")
            return job_id
            
        except Exception as e:
            logging.error(f"Pipeline: Failed to start training: {e}")
            raise
    
    def get_pipeline_status(self):
        """
        Get status of all pipeline components
        """
        try:
            logger_status = PipelineStatus.query.filter_by(component='logger').first()
            exporter_status = PipelineStatus.query.filter_by(component='exporter').first()
            trainer_status = PipelineStatus.query.filter_by(component='trainer').first()
            
            return {
                'logger': {
                    'status': logger_status.status if logger_status else 'unknown',
                    'last_activity': logger_status.last_activity.isoformat() if logger_status and logger_status.last_activity else None,
                    'message': logger_status.message if logger_status else None
                },
                'exporter': {
                    'status': exporter_status.status if exporter_status else 'unknown',
                    'last_activity': exporter_status.last_activity.isoformat() if exporter_status and exporter_status.last_activity else None,
                    'message': exporter_status.message if exporter_status else None
                },
                'trainer': {
                    'status': trainer_status.status if trainer_status else 'unknown',
                    'last_activity': trainer_status.last_activity.isoformat() if trainer_status and trainer_status.last_activity else None,
                    'message': trainer_status.message if trainer_status else None
                }
            }
            
        except Exception as e:
            logging.error(f"Failed to get pipeline status: {e}")
            return {
                'logger': {'status': 'error', 'message': str(e)},
                'exporter': {'status': 'error', 'message': str(e)},
                'trainer': {'status': 'error', 'message': str(e)}
            }
