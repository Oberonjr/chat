"""
Manages chat persistence - loading and saving conversations.
"""
import os
import json
import logging
from datetime import datetime
from app.config import CHATS_DIR

logger = logging.getLogger(__name__)

class ChatManager:
    """Handles chat persistence."""
    
    @staticmethod
    def get_chat_folder(character_name):
        """Get the chat folder path for a character."""
        folder_path = os.path.join(CHATS_DIR, character_name)
        os.makedirs(folder_path, exist_ok=True)
        return folder_path
    
    @staticmethod
    def generate_chat_id():
        """Generate a new chat ID based on timestamp."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    @staticmethod
    def save_chat(character_name, chat_id, chat_history):
        """
        Save chat history to a JSON file.
        
        Args:
            character_name: Name of the character
            chat_id: Unique chat identifier
            chat_history: List of conversation turns
        """
        folder_path = ChatManager.get_chat_folder(character_name)
        file_path = os.path.join(folder_path, f"{chat_id}.json")
        
        data = {
            "character": character_name,
            "chat_id": chat_id,
            "last_updated": datetime.now().isoformat(),
            "history": chat_history
        }
        
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved chat {chat_id} for {character_name}")
        except Exception as e:
            logger.error(f"Error saving chat: {e}")
    
    @staticmethod
    def load_chat(character_name, chat_id):
        """
        Load chat history from a JSON file.
        
        Args:
            character_name: Name of the character
            chat_id: Unique chat identifier
        
        Returns:
            List of conversation turns, or empty list if not found
        """
        folder_path = ChatManager.get_chat_folder(character_name)
        file_path = os.path.join(folder_path, f"{chat_id}.json")
        
        if not os.path.exists(file_path):
            logger.debug(f"Chat file not found: {file_path}")
            return []
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("history", [])
        except Exception as e:
            logger.error(f"Error loading chat: {e}")
            return []
    
    @staticmethod
    def list_chats(character_name):
        """
        List all chat IDs for a character.
        
        Args:
            character_name: Name of the character
        
        Returns:
            List of chat IDs (filenames without .json)
        """
        folder_path = ChatManager.get_chat_folder(character_name)
        
        try:
            files = os.listdir(folder_path)
            chat_ids = [f[:-5] for f in files if f.endswith(".json")]
            return sorted(chat_ids, reverse=True)  # Newest first
        except Exception as e:
            logger.error(f"Error listing chats: {e}")
            return []
    
    @staticmethod
    def delete_chat(character_name, chat_id):
        """
        Delete a chat file.
        
        Args:
            character_name: Name of the character
            chat_id: Unique chat identifier
        """
        folder_path = ChatManager.get_chat_folder(character_name)
        file_path = os.path.join(folder_path, f"{chat_id}.json")
        
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Deleted chat {chat_id}")
        except Exception as e:
            logger.error(f"Error deleting chat: {e}")

# Global instance
chat_manager = ChatManager()