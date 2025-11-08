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
from app.config import MODEL_PATH, MAX_SEQ_LEN

logger = logging.getLogger(__name__)

class ModelLoader:
    """Singleton model loader."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.config = None
            self.model = None
            self.tokenizer = None
            self.cache = None
            self.generator = None
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self._initialized = True
    
    def load_model(self):
        """Load the ExLlamaV2 model."""
        if self.model is not None:
            logger.info("Model already loaded")
            return
        
        logger.info(f"Loading model from {MODEL_PATH}")
        logger.info(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        
        self.config = ExLlamaV2Config(MODEL_PATH)
        self.model = ExLlamaV2(self.config)
        self.model.load()
        self.tokenizer = ExLlamaV2Tokenizer(self.config)
        self.cache = ExLlamaV2Cache(self.model, max_seq_len=MAX_SEQ_LEN, batch_size=1)
        self.generator = ExLlamaV2BaseGenerator(self.model, self.cache, self.tokenizer)
        
        logger.info("? Model loaded successfully")
    
    def get_sampler_settings(self, temperature=0.8, top_p=0.9, top_k=50, rep_penalty=1.1):
        """Create sampler settings."""
        settings = ExLlamaV2Sampler.Settings()
        settings.temperature = temperature
        settings.top_p = top_p
        settings.top_k = top_k
        settings.token_repetition_penalty = rep_penalty
        return settings

# Global instance
model_loader = ModelLoader()