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
                         rep_penalty=1.15, max_tokens=500):
        """
        Generate a response with conversation history.
        """
        # Build prompt with history
        prompt = prompt_builder.build_prompt_with_history(
            character_name, chat_history, model_type, CONTEXT_TOKENS
        )
        
        # DEBUG: Log the actual prompt
        logger.info("="*50)
        logger.info(f"MODEL TYPE: {model_type}")
        logger.info(f"FULL PROMPT:\n{prompt}")
        logger.info("="*50)
        
        # Get stop sequences
        stop_sequences = prompt_builder.get_stop_sequences(model_type, character_name)
        logger.info(f"Stop sequences: {stop_sequences}")
        
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
            logger.info("Starting generation...")
            generated = model_loader.generator.generate_simple(prompt, settings, max_tokens).strip()
            
            logger.info(f"RAW GENERATED TEXT (length={len(generated)}):\n{generated}")
            logger.info("="*50)
            
            # Apply stop sequences
            for stop in stop_sequences:
                if stop in generated:
                    logger.info(f"Found stop sequence: {stop}")
                    generated = generated.split(stop)[0]
            
            generated = generated.strip()
            logger.info(f"FINAL OUTPUT: {generated}")
            logger.info(f"Generated {len(tokenizer.encode(generated))} tokens")
            
            return generated
        
        except Exception as e:
            logger.error(f"Generation error: {e}", exc_info=True)
            return "I apologize, but I encountered an error generating a response."

# Global instance
text_generator = TextGenerator()