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
    def build_prompt_with_history(character_name, chat_history, model_type="llama2", max_tokens=3500):
        """
        Build a prompt that includes conversation history.
        """
        personality = character_manager.get_personality_prompt(character_name)
        
        # Build conversation history (skip first bot message if it's just a greeting)
        history_to_include = []
        for i, entry in enumerate(chat_history):
            # Skip the initial greeting message (no user input)
            if i == 0 and not entry.get("user"):
                continue
            # Skip entries where bot response is empty (currently being generated)
            if not entry.get("bot"):
                history_to_include.append(entry)
            else:
                history_to_include.append(entry)
        
        # Format based on model type
        if model_type == "llama2":
            # Llama-2-Chat format
            prompt = f"[INST] <<SYS>>\n{personality}\n<</SYS>>\n\n"
            
            # Add conversation history
            for i, entry in enumerate(history_to_include):
                user_msg = entry.get("user", "")
                bot_msg = entry.get("bot", "")
                
                if user_msg:
                    if i > 0:
                        # Not the first message, add proper formatting
                        prompt += f"<s>[INST] {user_msg} [/INST]"
                    else:
                        # First message, already have [INST] open
                        prompt += f"{user_msg} [/INST]"
                
                if bot_msg:
                    prompt += f" {bot_msg} </s>"
            
            # If last entry has no bot response, prompt is ready for generation
            # (already ends with [/INST])
        
        elif model_type == "chatml":
            prompt = f"<|im_start|>system\n{personality}<|im_end|>\n"
            for entry in history_to_include:
                if entry.get("user"):
                    prompt += f"<|im_start|>user\n{entry['user']}<|im_end|>\n"
                if entry.get("bot"):
                    prompt += f"<|im_start|>assistant\n{entry['bot']}<|im_end|>\n"
            prompt += "<|im_start|>assistant\n"
        
        elif model_type == "alpaca":
            prompt = f"{personality}\n\n"
            for entry in history_to_include:
                if entry.get("user"):
                    prompt += f"### Instruction:\n{entry['user']}\n\n"
                if entry.get("bot"):
                    prompt += f"### Response:\n{entry['bot']}\n\n"
            if history_to_include and not history_to_include[-1].get("bot"):
                prompt += "### Response:\n"
        
        elif model_type == "vicuna":
            prompt = f"{personality}\n\n"
            for entry in history_to_include:
                if entry.get("user"):
                    prompt += f"USER: {entry['user']}\n"
                if entry.get("bot"):
                    prompt += f"ASSISTANT: {entry['bot']}\n"
            prompt += "ASSISTANT: "
        
        else:
            # Simple format
            prompt = f"{personality}\n\n"
            for entry in history_to_include:
                if entry.get("user"):
                    prompt += f"User: {entry['user']}\n"
                if entry.get("bot"):
                    prompt += f"{character_name}: {entry['bot']}\n"
            prompt += f"{character_name}: "
        
        logger.debug(f"Built prompt ({len(prompt)} chars)")
        
        return prompt
    
    @staticmethod
    def get_stop_sequences(model_type, character_name):
        """Get stop sequences for the model type."""
        if model_type == "llama2":
            return ["</s>", "<s>[INST]", "[INST]"]
        elif model_type == "chatml":
            return ["<|im_end|>", "<|im_start|>"]
        elif model_type == "alpaca":
            return ["###"]
        elif model_type == "vicuna":
            return ["\nUSER:"]
        else:
            return ["\nUser:", f"\n{character_name}:"]

# Global instance
prompt_builder = PromptBuilder()