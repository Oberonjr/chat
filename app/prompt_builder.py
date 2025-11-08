"""
Builds prompts with conversation history and proper formatting.
"""
import logging
from app.character_manager import character_manager
from app.model_loader import model_loader

logger = logging.getLogger(__name__)

class PromptBuilder:
    """Builds prompts with conversation history."""
    
    @staticmethod
    def build_prompt_with_history(character_name, chat_history, model_type="llama2", max_tokens=1800):
        """
        Build a prompt that includes conversation history.
        
        Args:
            character_name: Name of the character
            chat_history: List of {"user": "...", "bot": "..."} dictionaries
            model_type: Prompt format type
            max_tokens: Maximum tokens to use for context
        
        Returns:
            Formatted prompt string
        """
        personality = character_manager.get_personality_prompt(character_name)
        examples = character_manager.get_example_dialogues(character_name)
        
        # Build conversation history
        history_text = ""
        for entry in chat_history:
            if entry.get("user"):
                history_text += f"User: {entry['user']}\n"
            if entry.get("bot"):
                history_text += f"{character_name}: {entry['bot']}\n"
        
        # Trim history if too long
        tokenizer = model_loader.tokenizer
        while len(tokenizer.encode(personality + history_text)) > max_tokens:
            # Remove oldest exchange
            if len(chat_history) > 1:
                chat_history.pop(0)
                history_text = ""
                for entry in chat_history:
                    if entry.get("user"):
                        history_text += f"User: {entry['user']}\n"
                    if entry.get("bot"):
                        history_text += f"{character_name}: {entry['bot']}\n"
            else:
                break
        
        # Format based on model type
        if model_type == "llama2":
            # Llama-2-Chat format with history
            prompt = f"[INST] <<SYS>>\n{personality}\n<</SYS>>\n\n{history_text}{character_name}: [/INST]"
        
        elif model_type == "chatml":
            # ChatML format with history
            prompt = f"<|im_start|>system\n{personality}<|im_end|>\n"
            for entry in chat_history:
                if entry.get("user"):
                    prompt += f"<|im_start|>user\n{entry['user']}<|im_end|>\n"
                if entry.get("bot"):
                    prompt += f"<|im_start|>assistant\n{entry['bot']}<|im_end|>\n"
            prompt += "<|im_start|>assistant\n"
        
        elif model_type == "alpaca":
            # Alpaca format
            prompt = f"### Instruction:\n{personality}\n\n### Conversation:\n{history_text}\n### Response:\n"
        
        elif model_type == "vicuna":
            # Vicuna format
            prompt = f"{personality}\n\n{history_text}{character_name}:"
        
        else:
            # Simple format
            prompt = f"{personality}\n{history_text}{character_name}:"
        
        return prompt
    
    @staticmethod
    def get_stop_sequences(model_type, character_name):
        """Get stop sequences for the model type."""
        if model_type == "llama2":
            return ["[INST]", "[/INST]", "User:", "\nUser:"]
        elif model_type == "chatml":
            return ["<|im_end|>", "<|im_start|>user"]
        elif model_type == "alpaca":
            return ["###", "### Instruction", "User:"]
        elif model_type == "vicuna":
            return ["USER:", "\nUSER:"]
        else:
            return ["User:", f"\n{character_name}:", "\nUser:"]

# Global instance
prompt_builder = PromptBuilder()