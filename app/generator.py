"""
Text generation with conversation history and proper stopping.
"""
import logging
from app.model_loader import model_loader
from app.prompt_builder import prompt_builder
from app.config import CONTEXT_TOKENS

logger = logging.getLogger(__name__)

class TextGenerator:
    """Handles text generation with history."""
    
    @staticmethod
    def generate_response(character_name, chat_history, model_type="llama2", 
                         temperature=0.8, top_p=0.9, top_k=50, 
                         rep_penalty=1.1, max_tokens=500):
        """
        Generate a response with conversation history.
        
        Args:
            character_name: Name of the character
            chat_history: List of conversation turns
            model_type: Prompt format type
            temperature, top_p, top_k, rep_penalty: Sampling parameters
            max_tokens: Maximum tokens to generate
        
        Returns:
            Generated text response
        """
        # Build prompt with history
        prompt = prompt_builder.build_prompt_with_history(
            character_name, chat_history, model_type, CONTEXT_TOKENS
        )
        
        # Get stop sequences
        stop_sequences = prompt_builder.get_stop_sequences(model_type, character_name)
        
        # Create sampler settings
        settings = model_loader.get_sampler_settings(temperature, top_p, top_k, rep_penalty)
        
        # Calculate available tokens
        tokenizer = model_loader.tokenizer
        prompt_tokens = len(tokenizer.encode(prompt))
        available_tokens = CONTEXT_TOKENS - prompt_tokens
        max_tokens = min(max_tokens, available_tokens)
        
        logger.info(f"Prompt tokens: {prompt_tokens}, Max response tokens: {max_tokens}")
        
        # Generate response
        try:
            generated = model_loader.generator.generate_simple(prompt, settings, max_tokens).strip()
            
            # Apply stop sequences
            for stop in stop_sequences:
                if stop in generated:
                    generated = generated.split(stop)[0]
            
            generated = generated.strip()
            logger.info(f"Generated {len(tokenizer.encode(generated))} tokens")
            
            return generated
        
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return "I apologize, but I encountered an error generating a response."

# Global instance
text_generator = TextGenerator()