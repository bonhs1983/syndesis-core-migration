import os

class Config:
    # Paths
    DATA_DIR = 'data'
    LOGS_DIR = os.path.join(DATA_DIR, 'logs')
    DATASETS_DIR = os.path.join(DATA_DIR, 'datasets')
    MODELS_DIR = os.path.join(DATA_DIR, 'models')
    
    # JSONL log file
    INTERACTIONS_LOG_FILE = os.path.join(LOGS_DIR, 'interactions.jsonl')
    
    # HuggingFace configuration
    HF_TOKEN = os.getenv('HF_TOKEN', 'default_token')
    HF_MODEL_NAME = os.getenv('HF_MODEL_NAME', 'microsoft/DialoGPT-small')
    
    # Training configuration
    DEFAULT_EPOCHS = 3
    DEFAULT_BATCH_SIZE = 4
    DEFAULT_LEARNING_RATE = 5e-5
    
    # Agent configuration
    AGENT_MODEL = os.getenv('AGENT_MODEL', 'gpt-3.5-turbo')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'default_key')
