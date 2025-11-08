"""
Global configuration settings for the chatbot.
"""
import os

# Base directory
BASE_DIR = os.path.dirname(__file__)

# Paths
MODELS_DIR = os.path.join(BASE_DIR, "../models")
CHARACTER_CONFIG_PATH = os.path.join(BASE_DIR, "../characters/character_config.json")
CHATS_DIR = os.path.join(BASE_DIR, "../chats")

# Default model (can be changed)
DEFAULT_MODEL = "Llama-2-7B-Chat_GPTQ"

# Model settings
MAX_SEQ_LEN = 4096  # Increased from 2048
CONTEXT_TOKENS = 3500  # More room for conversation

# Generation defaults
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 50
DEFAULT_REP_PENALTY = 1.15  # Slightly higher to prevent loops
DEFAULT_MAX_TOKENS = 500

# A1111 settings
A1111_PORT_RANGE = (7860, 7870)
A1111_IMAGE_WIDTH = 512
A1111_IMAGE_HEIGHT = 768

# Environment - Force offline mode
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

def get_available_models():
    """Get list of available model folders."""
    if not os.path.exists(MODELS_DIR):
        return []
    
    models = []
    for item in os.listdir(MODELS_DIR):
        model_path = os.path.join(MODELS_DIR, item)
        if os.path.isdir(model_path) and not item.startswith('.'):
            # Check if it has config.json (valid model folder)
            if os.path.exists(os.path.join(model_path, "config.json")):
                models.append(item)
    
    return sorted(models)

def get_model_path(model_name):
    """Get full path for a model by name."""
    return os.path.join(MODELS_DIR, model_name)