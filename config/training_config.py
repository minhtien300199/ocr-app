import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API configuration
API_CONFIG = {
    'openai_api_key': os.getenv('OPENAI_API_KEY'),
    'openai_org_id': os.getenv('OPENAI_ORG_ID'),
    'project_id': os.getenv('PROJECT_ID', 'proj_g3utYELQzj5kL2MmI90ue20v')  # Add project ID with default value
}

# Training configuration
TRAINING_CONFIG = {
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.001,
    'validation_split': 0.2,
    'image_size': (224, 224),  # Standard size for vision models
    'max_text_length': 512,    # Maximum text length for GPT processing
}

# Model configuration
MODEL_CONFIG = {
    'model_name': 'gpt-4o-mini',
    'api_key': API_CONFIG['openai_api_key'],
    'org_id': API_CONFIG['openai_org_id'],
    'project_id': API_CONFIG['project_id'],  # Add project ID to model config
    'temperature': 0.7,
    'max_tokens': 512,
    'top_p': 1.0,
    'frequency_penalty': 0.0,
    'presence_penalty': 0.0
}

# Image configuration
IMAGE_CONFIG = {
    'max_size': 2048,  # Maximum image dimension
    'quality': 'high',  # Image detail level (high/low)
}

# Dataset configuration
DATASET_CONFIG = {
    'train_data_dir': 'data/train',
    'validation_data_dir': 'data/validation',
    'languages': ['en', 'vi'],  # Supported languages
}

# Model paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATHS = {
    'save_dir': os.path.join(BASE_DIR, 'models'),
    'model_name': 'gpt4_ocr_model.pt',
    'best_model_path': os.path.join(BASE_DIR, 'models', 'best_model.pt')
} 