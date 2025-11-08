"""
Image generation using Automatic1111 API.
"""
import logging
import requests
from app.config import A1111_PORT_RANGE, A1111_IMAGE_WIDTH, A1111_IMAGE_HEIGHT

logger = logging.getLogger(__name__)

class ImageGenerator:
    """Handles image generation via A1111 API."""
    
    def __init__(self):
        self.api_url = None
    
    def find_api(self):
        """Find the Automatic1111 API by probing ports."""
        start_port, end_port = A1111_PORT_RANGE
        
        for port in range(start_port, end_port + 1):
            try:
                url = f"http://127.0.0.1:{port}/sdapi/v1/sd-models"
                response = requests.get(url, timeout=1)
                if response.status_code == 200 and isinstance(response.json(), list):
                    self.api_url = f"http://127.0.0.1:{port}"
                    logger.info(f"🟢 A1111 API found at port {port}")
                    return self.api_url
            except (requests.ConnectionError, requests.Timeout):
                continue
            except Exception as e:
                logger.debug(f"Error checking port {port}: {e}")
                continue
        
        raise RuntimeError(f"A1111 API not found on ports {start_port}-{end_port}")
    
    def generate_image(self, prompt, negative_prompt="", steps=30, cfg_scale=7):
        """
        Generate an image using the A1111 API.
        
        Args:
            prompt: Text prompt for image generation
            negative_prompt: Negative prompt
            steps: Number of sampling steps
            cfg_scale: CFG scale
        
        Returns:
            Base64 encoded image string
        """
        if self.api_url is None:
            self.find_api()
        
        payload = {
            "prompt": f"masterpiece, best quality, amazing quality, very aesthetic, {prompt}",
            "negative_prompt": negative_prompt,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "width": A1111_IMAGE_WIDTH,
            "height": A1111_IMAGE_HEIGHT,
            "save_images": True
        }
        
        try:
            response = requests.post(f"{self.api_url}/sdapi/v1/txt2img", json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            
            if "images" in result and result["images"]:
                logger.info("✅ Image generated successfully")
                return result["images"][0]
            else:
                logger.warning("A1111 response received but no images returned")
                raise RuntimeError("No image returned from A1111")
        
        except requests.Timeout:
            logger.error("A1111 request timed out")
            raise RuntimeError("Image generation timed out")
        except requests.RequestException as e:
            logger.error(f"A1111 request failed: {e}")
            raise RuntimeError(f"Image generation failed: {e}")
    
    def should_generate_image(self, text):
        """
        Check if the user message should trigger image generation.
        
        Args:
            text: User message
        
        Returns:
            Boolean indicating whether to generate an image
        """
        trigger_words = ["show me", "picture of", "draw", "image of", "generate image"]
        return any(word in text.lower() for word in trigger_words)

# Global instance
image_generator = ImageGenerator()