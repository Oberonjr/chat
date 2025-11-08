"""
Handles ExLlamaV2 model initialization.
"""
import os
import logging
import torch
import gc
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
    
    def unload_current_model(self):
        """Safely unload the current model and free resources."""
        if self.model is None:
            return
        
        logger.info(f"Unloading model: {self.current_model_name}")
        
        try:
            # Delete in reverse order of creation
            if self.generator is not None:
                del self.generator
                self.generator = None
            
            if self.cache is not None:
                del self.cache
                self.cache = None
            
            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
            
            if self.model is not None:
                del self.model
                self.model = None
            
            if self.config is not None:
                del self.config
                self.config = None
            
            # Force garbage collection
            gc.collect()
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            logger.info("✅ Model unloaded successfully")
            
        except Exception as e:
            logger.error(f"Error during model unload: {e}", exc_info=True)
    
    def load_model(self, model_name):
        """Load or switch to a different model."""
        # If same model is already loaded, skip
        if self.model is not None and self.current_model_name == model_name:
            logger.info(f"Model '{model_name}' already loaded")
            return
        
        model_path = get_model_path(model_name)
        
        # Verify model path exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        
        # Check for required files
        config_file = os.path.join(model_path, "config.json")
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"config.json not found in {model_path}")
        
        logger.info(f"Loading model from {model_path}")
        logger.info(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # Unload previous model if exists
        self.unload_current_model()
        
        # Small delay to ensure resources are freed
        import time
        time.sleep(0.5)
        
        try:
            # Load new model
            logger.info("Creating config...")
            self.config = ExLlamaV2Config(model_path)
            
            # Set reasonable max sequence length
            self.config.max_seq_len = min(MAX_SEQ_LEN, 4096)
            logger.info(f"Max sequence length set to: {self.config.max_seq_len}")
            
            logger.info("Loading model weights...")
            self.model = ExLlamaV2(self.config)
            
            # Load without lazy loading
            logger.info("Loading model to device...")
            self.model.load()
            
            logger.info("Loading tokenizer...")
            self.tokenizer = ExLlamaV2Tokenizer(self.config)
            
            logger.info("Creating cache...")
            self.cache = ExLlamaV2Cache(
                self.model, 
                max_seq_len=self.config.max_seq_len, 
                batch_size=1
            )
            
            logger.info("Creating generator...")
            self.generator = ExLlamaV2BaseGenerator(self.model, self.cache, self.tokenizer)
            
            self.current_model_name = model_name
            logger.info(f"✅ Model '{model_name}' loaded successfully")
            
            # Test tokenization
            test_text = "Hello, how are you?"
            test_tokens = self.tokenizer.encode(test_text)
            logger.info(f"Tokenizer test: '{test_text}' -> {test_tokens.shape if hasattr(test_tokens, 'shape') else len(test_tokens)} tokens")
            
        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {e}", exc_info=True)
            # Clean up partial load
            self.unload_current_model()
            self.current_model_name = None
            raise RuntimeError(f"Model loading failed: {str(e)}")
    
    def get_sampler_settings(self, temperature=0.8, top_p=0.9, top_k=50, rep_penalty=1.15):
        """Create sampler settings."""
        settings = ExLlamaV2Sampler.Settings()
        settings.temperature = temperature
        settings.top_p = top_p
        settings.top_k = top_k
        settings.token_repetition_penalty = rep_penalty
        
        logger.debug(f"Sampler settings: temp={temperature}, top_p={top_p}, top_k={top_k}, rep_penalty={rep_penalty}")
        
        return settings

# Global instance
model_loader = ModelLoader()