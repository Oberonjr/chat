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
        
        # Build conversation history (skip first bot message if it's greeting)
        history_to_include = []
        for i, entry in enumerate(chat_history):
            # Skip the initial greeting message
            if i == 0 and not entry.get("user"):
                continue
            history_to_include.append(entry)
        
        # Format based on model type
        if model_type == "llama2":
            # Llama-2-Chat CORRECT format
            if len(history_to_include) == 0:
                # First user message
                prompt = f"[INST] <<SYS>>\n{personality}\n<</SYS>>\n\n [/INST]"
            else:
                # Build conversation with proper structure
                prompt = f"<s>[INST] <<SYS>>\n{personality}\n<</SYS>>\n\n"
                
                for i, entry in enumerate(history_to_include):
                    user_msg = entry.get("user", "")
                    bot_msg = entry.get("bot", "")
                    
                    if i == 0:
                        # First exchange
                        prompt += f"{user_msg} [/INST]"
                        if bot_msg:
                            prompt += f" {bot_msg} </s>"
                    else:
                        # Subsequent exchanges
                        if user_msg:
                            prompt += f"<s>[INST] {user_msg} [/INST]"
                        if bot_msg:
                            prompt += f" {bot_msg} </s>"
                
                # If last entry has no bot response, we're generating it now
                if history_to_include and not history_to_include[-1].get("bot"):
                    # Already ended with [/INST], ready for generation
                    pass
                else:
                    # Need to add a new [INST] for the next user message
                    # This shouldn't happen in generation flow
                    pass
        
        elif model_type == "chatml":
            # ChatML format
            prompt = f"<|im_start|>system\n{personality}<|im_end|>\n"
            for entry in history_to_include:
                if entry.get("user"):
                    prompt += f"<|im_start|>user\n{entry['user']}<|im_end|>\n"
                if entry.get("bot"):
                    prompt += f"<|im_start|>assistant\n{entry['bot']}<|im_end|>\n"
            prompt += "<|im_start|>assistant\n"
        
        elif model_type == "alpaca":
            # Alpaca format
            history_text = ""
            for entry in history_to_include:
                if entry.get("user"):
                    history_text += f"### Input:\n{entry['user']}\n"
                if entry.get("bot"):
                    history_text += f"### Response:\n{entry['bot']}\n\n"
            
            prompt = f"### Instruction:\n{personality}\n\n{history_text}### Response:\n"
        
        elif model_type == "vicuna":
            # Vicuna format
            prompt = f"A chat between a curious user and an assistant. {personality}\n\n"
            for entry in history_to_include:
                if entry.get("user"):
                    prompt += f"USER: {entry['user']}\n"
                if entry.get("bot"):
                    prompt += f"ASSISTANT: {entry['bot']}\n"
            prompt += "ASSISTANT:"
        
        else:
            # Simple format
            prompt = f"{personality}\n\n"
            for entry in history_to_include:
                if entry.get("user"):
                    prompt += f"User: {entry['user']}\n"
                if entry.get("bot"):
                    prompt += f"{character_name}: {entry['bot']}\n"
            prompt += f"{character_name}:"
        
        # Trim if too long
        tokenizer = model_loader.tokenizer
        if tokenizer:
            tokens = tokenizer.encode(prompt)
            if len(tokens) > max_tokens:
                logger.warning(f"Prompt too long ({len(tokens)} tokens), trimming...")
                # Remove oldest messages
                while len(tokens) > max_tokens and len(history_to_include) > 1:
                    history_to_include.pop(0)
                    # Rebuild prompt
                    prompt = PromptBuilder.build_prompt_with_history(
                        character_name, [{"user": "", "bot": ""}] + history_to_include, 
                        model_type, max_tokens
                    )
                    tokens = tokenizer.encode(prompt)
        
        return prompt
    
    @staticmethod
    def get_stop_sequences(model_type, character_name):
        """Get stop sequences for the model type."""
        if model_type == "llama2":
            return ["</s>", "[INST]", "<s>[INST]"]
        elif model_type == "chatml":
            return ["<|im_end|>", "<|im_start|>"]
        elif model_type == "alpaca":
            return ["###", "### Input:", "### Instruction:"]
        elif model_type == "vicuna":
            return ["USER:", "\nUSER:"]
        else:
            return ["\nUser:", f"\n{character_name}:", "\n\n"]

# Global instance
prompt_builder = PromptBuilder()