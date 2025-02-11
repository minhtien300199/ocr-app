import os
import torch

# Training configuration
TRAINING_CONFIG = {
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.001,
    'validation_split': 0.2,
    'image_size': (224, 224),  # Standard size for vision transformers
    'max_text_length': 512,    # Maximum text length for GPT processing
}

# Model configuration
MODEL_CONFIG = {
    'model_name': 'microsoft/gpt-4-vision-mini',  # or your preferred vision-language model
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'temperature': 0.7,
    'max_new_tokens': 512,
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