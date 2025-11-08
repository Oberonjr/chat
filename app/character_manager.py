"""
Manages character configurations and loading.
"""
import json
import logging
from app.config import CHARACTER_CONFIG_PATH

logger = logging.getLogger(__name__)

class CharacterManager:
    """Manages character cards and configurations."""
    
    def __init__(self):
        self.characters = {}
        self.load_characters()
    
    def load_characters(self):
        """Load character configurations from JSON."""
        try:
            with open(CHARACTER_CONFIG_PATH, "r", encoding="utf-8") as f:
                self.characters = json.load(f)
            logger.info(f"Loaded {len(self.characters)} characters")
        except FileNotFoundError:
            logger.error(f"Character config not found: {CHARACTER_CONFIG_PATH}")
            self.characters = {}
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing character config: {e}")
            self.characters = {}
    
    def get_character(self, name):
        """Get a character by name."""
        return self.characters.get(name, {})
    
    def get_character_names(self):
        """Get list of all character names."""
        return list(self.characters.keys())
    
    def get_personality_prompt(self, name):
        """Get the personality prompt for a character."""
        char = self.get_character(name)
        return char.get("personality_prompt", "")
    
    def get_first_message(self, name):
        """Get the first message for a character."""
        char = self.get_character(name)
        return char.get("first_message", "Hello! How can I help you today?")
    
    def get_example_dialogues(self, name):
        """Get example dialogues for a character."""
        char = self.get_character(name)
        return char.get("example_dialogues", [])

# Global instance
character_manager = CharacterManager()