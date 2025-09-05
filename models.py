from app import db
from datetime import datetime
from sqlalchemy import Text, Integer, String, DateTime, Boolean, Float

class InteractionLog(db.Model):
    id = db.Column(Integer, primary_key=True)
    timestamp = db.Column(DateTime, default=datetime.utcnow, nullable=False)
    agent_input = db.Column(Text, nullable=False)
    agent_output = db.Column(Text, nullable=False)
    context = db.Column(Text)
    session_id = db.Column(String(64))
    processed = db.Column(Boolean, default=False)
    
class TrainingJob(db.Model):
    id = db.Column(Integer, primary_key=True)
    created_at = db.Column(DateTime, default=datetime.utcnow, nullable=False)
    status = db.Column(String(32), default='pending')  # pending, running, completed, failed
    dataset_path = db.Column(String(256))
    model_output_path = db.Column(String(256))
    log_output = db.Column(Text)
    error_message = db.Column(Text)
    total_samples = db.Column(Integer, default=0)
    epochs = db.Column(Integer, default=1)
    
class PipelineStatus(db.Model):
    id = db.Column(Integer, primary_key=True)
    component = db.Column(String(64), nullable=False)  # logger, exporter, trainer
    status = db.Column(String(32), default='idle')  # idle, running, error
    last_activity = db.Column(DateTime, default=datetime.utcnow)
    message = db.Column(Text)

class TraitState(db.Model):
    id = db.Column(Integer, primary_key=True)
    session_id = db.Column(String(64), nullable=False)
    timestamp = db.Column(DateTime, default=datetime.utcnow, nullable=False)
    empathy = db.Column(Float, default=0.0)
    creativity = db.Column(Float, default=0.0)
    humor = db.Column(Float, default=0.0)
    curiosity = db.Column(Float, default=0.0)
    supportiveness = db.Column(Float, default=0.0)
    analyticalness = db.Column(Float, default=0.0)
