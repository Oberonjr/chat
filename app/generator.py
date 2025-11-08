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
        
        # Get stop sequences
        stop_sequences = prompt_builder.get_stop_sequences(model_type, character_name)
        
        # Create sampler settings
        settings = model_loader.get_sampler_settings(temperature, top_p, top_k, rep_penalty)
        
        # Calculate available tokens
        tokenizer = model_loader.tokenizer
        
        try:
            encoded = tokenizer.encode(prompt)
            if hasattr(encoded, 'shape'):
                prompt_tokens = encoded.shape[-1]
            else:
                prompt_tokens = len(encoded)
        except Exception as e:
            logger.error(f"Tokenizer encoding error: {e}")
            prompt_tokens = 100
        
        available_tokens = CONTEXT_TOKENS - prompt_tokens
        max_tokens = min(max_tokens, available_tokens, 1000)
        
        if max_tokens < 10:
            max_tokens = 50
        
        logger.info(f"Prompt tokens: {prompt_tokens}, Max response tokens: {max_tokens}")
        
        # Generate response
        try:
            logger.info(f"Starting generation...")
            
            import time
            start_time = time.time()
            
            # Generate
            generated = model_loader.generator.generate_simple(
                prompt,
                settings,
                max_tokens,
                encode_special_tokens=True
            )
            
            elapsed = time.time() - start_time
            logger.info(f"Generation completed in {elapsed:.2f} seconds")
            
            # CRITICAL: Remove the prompt from the generated text
            # ExLlama sometimes returns prompt + generation
            if generated.startswith(prompt):
                generated = generated[len(prompt):]
            
            # Clean up
            output_text = generated.strip()
            
            logger.info(f"RAW OUTPUT (first 500 chars):\n{output_text[:500]}")
            
            # Apply stop sequences
            for stop in stop_sequences:
                if stop in output_text:
                    logger.info(f"Found stop sequence: '{stop}'")
                    # Take everything BEFORE the stop sequence
                    parts = output_text.split(stop)
                    output_text = parts[0].strip()
                    break
            
            # Additional cleaning
            # Remove any remaining special tokens
            output_text = output_text.replace("</s>", "").replace("<s>", "").strip()
            
            # Remove "System:" lines if they appear
            lines = output_text.split("\n")
            cleaned_lines = [line for line in lines if not line.strip().startswith("System:")]
            output_text = "\n".join(cleaned_lines).strip()
            
            # If output is empty or too short, provide fallback
            if len(output_text) < 5:
                logger.warning(f"Generated text too short: '{output_text}'")
                return "Hello! How can I assist you today?"
            
            logger.info(f"FINAL OUTPUT ({len(output_text)} chars): {output_text[:200]}...")
            
            return output_text
        
        except Exception as e:
            logger.error(f"Generation error: {e}", exc_info=True)
            return "I apologize, but I encountered an error generating a response."

# Global instance
text_generator = TextGenerator()