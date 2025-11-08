"""
Global configuration settings for the chatbot.
"""
import os

# Base directory
BASE_DIR = os.path.dirname(__file__)

# Paths
MODEL_PATH = os.path.join(BASE_DIR, "../models/Llama-2-7B-Chat_GPTQ")
CHARACTER_CONFIG_PATH = os.path.join(BASE_DIR, "../characters/character_config.json")
CHATS_DIR = os.path.join(BASE_DIR, "../chats")

# Model settings
MAX_SEQ_LEN = 2048
CONTEXT_TOKENS = 1800  # Reserve space for response

# Generation defaults
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 50
DEFAULT_REP_PENALTY = 1.1
DEFAULT_MAX_TOKENS = 500

# A1111 settings
A1111_PORT_RANGE = (7860, 7870)
A1111_IMAGE_WIDTH = 512
A1111_IMAGE_HEIGHT = 768

# Environment
os.environ["TRANSFORMERS_OFFLINE"] = "1"