"""
Handles ExLlamaV2 model initialization.
"""
import logging
import torch
from exllamav2 import ExLlamaV2, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2BaseGenerator
from exllamav2.generator.sampler import ExLlamaV2Sampler
from exllamav2.config import ExLlamaV2Config
from exllamav2.cache import ExLlamaV2Cache
from app.config import get_model_path, MAX_SEQ_LEN

logger = logging.getLogger(__name__)

class ModelLoader:
    """Singleton model loader with dynamic model switching."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.config = None
            self.model = None
            self.tokenizer = None
            self.cache = None
            self.generator = None
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.current_model_name = None
            self.initialized = True
    
    def load_model(self, model_name):
        """Load or switch to a different model."""
        # If same model is already loaded, skip
        if self.model is not None and self.current_model_name == model_name:
            logger.info(f"Model '{model_name}' already loaded")
            return
        
        model_path = get_model_path(model_name)
        logger.info(f"Loading model from {model_path}")
        logger.info(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        
        # Unload previous model if exists
        if self.model is not None:
            logger.info(f"Unloading previous model: {self.current_model_name}")
            del self.model
            del self.cache
            del self.generator
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Load new model
        self.config = ExLlamaV2Config(model_path)
        
        # Set max sequence length
        self.config.max_seq_len = MAX_SEQ_LEN
        
        self.model = ExLlamaV2(self.config)
        self.model.load()
        self.tokenizer = ExLlamaV2Tokenizer(self.config)
        self.cache = ExLlamaV2Cache(self.model, max_seq_len=MAX_SEQ_LEN, batch_size=1)
        self.generator = ExLlamaV2BaseGenerator(self.model, self.cache, self.tokenizer)
        
        self.current_model_name = model_name
        logger.info(f"✅ Model '{model_name}' loaded successfully")
    
    def get_sampler_settings(self, temperature=0.8, top_p=0.9, top_k=50, rep_penalty=1.15):
        """Create sampler settings."""
        settings = ExLlamaV2Sampler.Settings()
        settings.temperature = temperature
        settings.top_p = top_p
        settings.top_k = top_k
        settings.token_repetition_penalty = rep_penalty
        
        # Add min_p for better quality
        settings.min_p = 0.05
        
        return settings

# Global instance
model_loader = ModelLoader()