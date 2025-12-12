"""
Configuration management for the application
"""

import os
from pathlib import Path
from typing import Optional

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
CLEANED_DATA_DIR = DATA_DIR / 'cleaned'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
RAW_DATA_DIR.mkdir(exist_ok=True)
CLEANED_DATA_DIR.mkdir(exist_ok=True)
PROCESSED_DATA_DIR.mkdir(exist_ok=True)


class Config:
    """Application configuration"""
    
    # Kafka Configuration
    KAFKA_BOOTSTRAP_SERVERS = os.getenv(
        'KAFKA_BOOTSTRAP_SERVERS', 
        'localhost:9092'
    ).split(',')
    
    # Producer Configuration
    PRODUCER_PORT = int(os.getenv('PRODUCER_PORT', '5000'))
    PRODUCER_URL = os.getenv('PRODUCER_URL', f'http://127.0.0.1:5000')
    
    # Reddit Configuration
    REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID', 'MASKED_VISHAL SRIVASTAVA')
    REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET', 'MASKED_VISHAL SRIVASTAVA')
    REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT', 'realtime-streamer')
    
    # Consumer Configuration
    KAFKA_TOPIC = os.getenv('KAFKA_TOPIC', 'redditStream')
    DATA_FILE = os.getenv('DATA_FILE', 'reddit_data.csv')
    DATA_FILE_PATH = DATA_DIR / DATA_FILE
    
    # API Configuration
    API_PORT = int(os.getenv('API_PORT', '5001'))
    API_URL = os.getenv('API_URL', f'http://localhost:5001')
    
    # TRIAGE Agent Configuration
    TRIAGE_OUTPUT_DIR = CLEANED_DATA_DIR
    
    # REPAIR Agent Configuration
    REPAIR_OUTPUT_DIR = PROCESSED_DATA_DIR
    
    # Prediction Service Configuration
    MODEL_CONFIG_FILE = BASE_DIR / 'config' / 'model_config.json'
    DEFAULT_MODEL_PATH = os.getenv('MODEL_PATH', 'unitary/toxic-bert')
    CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.8'))
    
    # ADAPT Agent Configuration
    ADAPT_OUTPUT_DIR = BASE_DIR / 'models'
    LORA_R = int(os.getenv('LORA_R', '8'))
    LORA_ALPHA = int(os.getenv('LORA_ALPHA', '16'))
    
    # Ollama Configuration (for REPAIR agent)
    OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    OLLAMA_REPAIR_MODEL = os.getenv('OLLAMA_REPAIR_MODEL', 'llama3:8b-instruct')
    
    # Subreddit Configuration
    SUBREDDIT = os.getenv('SUBREDDIT', 'UCI')
    LIMIT = int(os.getenv('LIMIT', '200'))
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    @classmethod
    def get_kafka_servers(cls) -> list:
        """Get Kafka bootstrap servers as list"""
        return cls.KAFKA_BOOTSTRAP_SERVERS
    
    @classmethod
    def validate(cls) -> bool:
        """Validate configuration"""
        # Add validation logic here
        return True

