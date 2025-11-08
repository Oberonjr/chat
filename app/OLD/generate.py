# =========================
# Imports & Environment Setup
# =========================

import os
import json
import requests
import random
import time
import datetime
import threading
import logging

import torch
import gradio as gr

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Get the base directory of the script
BASE_DIR = os.path.dirname(__file__)  # Directory containing generate.py

# =========================
# Model & Character Configuration
# =========================

# Dynamically construct paths relative to BASE_DIR
model_path = os.path.join(BASE_DIR, "../models/Llama-2-7B-Chat_GPTQ")
character_config_path = os.path.join(BASE_DIR, "../characters/character_config.json")

os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Check if model files exist
if not os.path.exists(model_path):
    logging.warning(f"Model path '{model_path}' does not exist. Ensure all required files are available offline.")

# Print model files and device info for debugging
logging.info("Files in model path: %s", os.listdir(model_path) if os.path.exists(model_path) else "Path not found")
logging.info("CUDA Device: %s", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
logging.info("CUDA Available: %s", torch.cuda.is_available())
logging.info("MPS Available: %s", torch.backends.mps.is_available())
logging.info("Torch Version: %s", torch.version)

# Load character configurations
with open(character_config_path, "r", encoding="utf-8") as f:
    characters = json.load(f)
character_names = list(characters.keys())

# =========================
# LLaMA Model Initialization (ExLLaMA V2)
# =========================

from exllamav2 import ExLlamaV2, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2BaseGenerator
from exllamav2.generator.sampler import ExLlamaV2Sampler
from exllamav2.config import ExLlamaV2Config
from exllamav2.cache import ExLlamaV2Cache

config = ExLlamaV2Config(model_path)
model = ExLlamaV2(config)
model.load()
tokenizer = ExLlamaV2Tokenizer(config)
cache = ExLlamaV2Cache(model, max_seq_len=2048, batch_size=1)
generator = ExLlamaV2BaseGenerator(model, cache, tokenizer)

# Default sampling settings
settings = ExLlamaV2Sampler.Settings()
settings.token_repetition_penalty = 1.1
settings.temperature = 0.8
settings.top_p = 0.9
settings.top_k = 50

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Chatbot Logic & State
# =========================

current_chat_id = "12345678"  # Static ID for testing
chat_log = []
stream_state = {"text": ""}  # Initialize stream_state to store the generated text
done_event = threading.Event()  # Use threading.Event for signaling
stream_lock = threading.Lock()  # Add a lock for thread-safe updates

# =========================
# Utility Functions
# =========================

def find_a1111_api(start_port=7860, end_port=7870):
    """Finds the Automatic1111 API port by probing a range."""
    for port in range(start_port, end_port + 1):
        try:
            url = f"http://127.0.0.1:{port}/sdapi/v1/sd-models"
            response = requests.get(url, timeout=1)
            if response.status_code == 200 and isinstance(response.json(), list):
                logging.info(f"🟢 A1111 API found at port {port}")
                return f"http://127.0.0.1:{port}"
        except requests.ConnectionError:
            logging.debug(f"Connection error on port {port}")
        except requests.Timeout:
            logging.debug(f"Timeout on port {port}")
        except requests.RequestException as e:
            logging.debug(f"Request exception on port {port}: {e}")
    raise RuntimeError("Automatic1111 API not found on ports 7860-7870")

def generate_image(prompt):
    """Generates an image using the Automatic1111 API and returns base64."""
    payload = {
        "prompt": "masterpiece, best quality, amazing quality, very aesthetic" + prompt,
        "steps": 30,
        "cfg_scale": 7,
        "width": 512,
        "height": 768,
        "save_images": True
    }

    try:
        A1111_BASE = find_a1111_api()
        response = requests.post(f"{A1111_BASE}/sdapi/v1/txt2img", json=payload)
        response.raise_for_status()
        r = response.json()
        if "images" in r and r["images"]:
            return r["images"][0]  # Already a base64 string
        logging.warning("A1111 response received but no images returned: %s", r)
        raise RuntimeError("Image generation failed or no image returned.")
    except requests.ConnectionError as e:
        logging.error("Connection error from A1111: %s", e)
        raise RuntimeError("Image generation HTTP request failed.")
    except requests.Timeout as e:
        logging.error("Timeout error from A1111: %s", e)
        raise RuntimeError("Image generation HTTP request failed.")
    except Exception as e:
        logging.error("Image generation error: %s", e)
        raise

def format_chat_html(chat_log):
    """Formats the chat log as HTML for display."""
    html = ""
    for entry in chat_log:
        user_msg = entry.get("user", "")
        bot_msg = entry.get("bot", "")
        image_b64 = entry.get("image", None)
        if user_msg:
            html += f"""
            <div style='text-align: right; margin-bottom: 6px;'>
                <div style='display: inline-block; background-color: rgba(135, 206, 250, 0.3); 
                            padding: 10px 14px; border-radius: 18px; max-width: 70%; font-family: sans-serif;'>
                    {user_msg}
                </div>
            </div>
            """
        if bot_msg:
            html += f"""
            <div style='text-align: left; margin-bottom: 6px;'>
                <div style='display: inline-block; background-color: rgba(200, 200, 200, 0.3); 
                            padding: 10px 14px; border-radius: 18px; max-width: 70%; font-family: sans-serif;'>
                    {bot_msg}
                </div>
            </div>
            """
        if image_b64:
            html += f"""
            <div style='text-align: left; margin: 10px 0 20px 0;'>
                <img src="data:image/png;base64,{image_b64}" 
                     style="max-width: 256px; border-radius: 8px;" />
            </div>
            """
    return html

def save_chat_log(character_name, chat_id, chat_log):
    """Saves the chat log to a JSON file per character."""
    folder_path = os.path.join(BASE_DIR, "../chats", character_name)  # Dynamically construct path
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, f"{chat_id}.json")
    if not os.path.exists(file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump({"character": character_name, "chat_id": chat_id, "history": []}, f, indent=2, ensure_ascii=False)
    with open(file_path, "r+", encoding="utf-8") as f:
        data = json.load(f)
        data["history"] = chat_log
        f.seek(0)
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.truncate()

# =========================
# Chatbot Response Generation
# =========================

def build_prompt(character_name, input_text, model_type="llama2"):
    """Builds the correct prompt format based on model type."""
    personality_prompt = characters[character_name]["personality_prompt"]
    
    if model_type == "llama2":
        # Llama-2-Chat format
        return f"[INST] <<SYS>>\n{personality_prompt}\n<</SYS>>\n\n{input_text} [/INST]"
    
    elif model_type == "chatml":
        # ChatML format (used by Nous-Hermes, some Dolphin models)
        return f"<|im_start|>system\n{personality_prompt}<|im_end|>\n<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n"
    
    elif model_type == "alpaca":
        # Alpaca format (used by WizardLM-Uncensored)
        return f"### Instruction:\n{personality_prompt}\n\n### Input:\n{input_text}\n\n### Response:\n"
    
    elif model_type == "vicuna":
        # Vicuna format (used by some models)
        return f"{personality_prompt}\n\nUSER: {input_text}\nASSISTANT:"
    
    else:
        # Simple format (fallback)
        return f"{personality_prompt}\nUser: {input_text}\n{character_name}:"

def threaded_response_generation(input_text, character_name, temperature, top_p, top_k, rep_penalty, max_tokens, model_type="llama2"):
    """Handles response generation in a separate thread."""
    # Use the new build_prompt function
    prompt = build_prompt(character_name, input_text, model_type)
    
    settings.temperature = temperature
    settings.top_p = top_p
    settings.top_k = top_k
    settings.token_repetition_penalty = rep_penalty

    # Calculate the number of tokens in the prompt
    prompt_tokens = len(tokenizer.encode(prompt))
    max_tokens = min(max_tokens, 2048 - prompt_tokens)

    # Reset thread state
    with stream_lock:
        stream_state["text"] = ""
    done_event.clear()

    # Add user message to chat log
    chat_log.append({"user": input_text, "bot": ""})

    def run_generation():
        """Thread worker for generating the response."""
        try:
            # Check for trigger words for image generation
            trigger_words = ["show me", "picture of", "draw", "image of"]
            image_b64 = None
            if any(word in input_text.lower() for word in trigger_words):
                try:
                    image_b64 = generate_image(input_text)
                except Exception as e:
                    logging.error(f"Image generation failed: {e}")
                    with stream_lock:
                        stream_state["text"] += "\n(Side note: image generation failed.)"

            # Generate text response
            generated = generator.generate_simple(prompt, settings, max_tokens).strip()
            
            # Clean up response (remove prompt echoes or special tokens)
            generated = generated.split("[INST]")[0]  # Remove any echoed instructions
            generated = generated.split("<|im_end|>")[0]  # Remove ChatML end tokens
            generated = generated.strip()
            
            # Simulate streaming by splitting the response into chunks
            for i in range(0, len(generated), 50):
                with stream_lock:
                    stream_state["text"] = generated[:i + 50]
                time.sleep(0.1)

            # Update the last bot message in the chat log
            chat_log[-1]["bot"] = stream_state["text"]

            # Add image to the last bot message if generated
            if image_b64:
                chat_log[-1]["image"] = image_b64

            # Save chat log
            save_chat_log(character_name, current_chat_id, chat_log)
        except Exception as e:
            logging.error(f"Error during generation: {e}")
        finally:
            done_event.set()

    # Start the thread
    threading.Thread(target=run_generation).start()

def handle_submit(input_text, character, model_type, temperature, top_p, top_k, rep_penalty, max_tokens, stream_response):
    """Handles the submit button click in Gradio UI."""
    # Pass model_type to threaded_response_generation
    threaded_response_generation(input_text, character, temperature, top_p, top_k, rep_penalty, max_tokens, model_type)

    if stream_response:
        while not done_event.is_set():
            yield format_chat_html(chat_log)
            time.sleep(0.1)
        yield format_chat_html(chat_log)
    else:
        while not done_event.is_set():
            time.sleep(0.1)
        yield format_chat_html(chat_log)

# =========================
# Gradio UI Setup
# =========================

with gr.Blocks(css="""
#chat-area {
    height: calc(100vh - 220px);
    overflow-y: auto;
    padding: 10px;
    border: 1px solid #ccc;
    border-radius: 8px;
}
#input-container {
    display: flex;
    margin-top: auto;
    width: 100%;
}
#textbox {
    flex-grow: 1;
    height: 60px !important;
    padding: 6px;
    margin: 0;
}
#send-button {
    width: 120px;
    margin-left: 6px;
    margin-right: 0;
    padding: 6px 12px;
}
""") as demo:

    gr.Markdown("# 🤖 LLaMA Chatbot (Character + Image Gen)")

    with gr.Row():
        with gr.Column(scale=3):
            chat_area = gr.HTML(label="Conversation", elem_id="chat-area")  # Use HTML for dynamic updates
            with gr.Row(elem_id="input-container"):
                input_box = gr.Textbox(lines=2, placeholder="Type your message...", elem_id="textbox", show_label=False)
                submit = gr.Button("Send", elem_id="send-button")

        with gr.Column(scale=1):
            character = gr.Dropdown(character_names, label="Select Character", value=character_names[0])
            model_type = gr.Dropdown(
                ["llama2", "chatml", "alpaca", "vicuna", "simple"],
                label="Prompt Format",
                value="llama2",
                info="Must match your model type"
            )
            stream_checkbox = gr.Checkbox(label="Stream Response", value=True)
            gr.Markdown("### Response Settings")
            max_tokens_slider = gr.Slider(32, 2048, value=500, step=8, label="Max Tokens")  # Adjusted max value
            temp_slider = gr.Slider(0.1, 1.5, value=0.8, label="Temperature")
            top_p_slider = gr.Slider(0.1, 1.0, value=0.9, label="Top P")
            top_k_slider = gr.Slider(0, 100, value=50, step=1, label="Top K")
            rep_penalty_slider = gr.Slider(1.0, 2.0, value=1.1, step=0.01, label="Repetition Penalty")

        submit.click(
            handle_submit,
            inputs=[
                input_box, character, model_type,  # Add model_type here
                temp_slider, top_p_slider, top_k_slider,
                rep_penalty_slider, max_tokens_slider,
                stream_checkbox
            ],
            outputs=chat_area,
)

# =========================
# App Launch
# =========================

demo.launch(share=False, inbrowser=True)
