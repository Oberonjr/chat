"""
Main Gradio UI for the chatbot with conversation history support.
"""
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
import threading
import time
import gradio as gr
from app.config import *
from app.model_loader import model_loader
from app.character_manager import character_manager
from app.chat_manager import chat_manager
from app.generator import text_generator
from app.image_gen import image_generator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# =========================
# Global State
# =========================

current_character = None
current_chat_id = None
current_model = None
chat_history = []
stream_state = {"text": ""}
done_event = threading.Event()
stream_lock = threading.Lock()

# =========================
# Chat Functions
# =========================

def format_chat_html(history):
    """Format chat history as HTML."""
    html = ""
    for entry in history:
        user_msg = entry.get("user", "")
        bot_msg = entry.get("bot", "")
        image_b64 = entry.get("image", None)
        
        if user_msg:
            html += f"""
            <div style='text-align: right; margin-bottom: 6px;'>
                <div style='display: inline-block; background-color: rgba(135, 206, 250, 0.3); 
                            padding: 10px 14px; border-radius: 18px; max-width: 70%; 
                            font-family: sans-serif; word-wrap: break-word;'>
                    {user_msg}
                </div>
            </div>
            """
        
        if bot_msg:
            html += f"""
            <div style='text-align: left; margin-bottom: 6px;'>
                <div style='display: inline-block; background-color: rgba(200, 200, 200, 0.3); 
                            padding: 10px 14px; border-radius: 18px; max-width: 70%; 
                            font-family: sans-serif; word-wrap: break-word;'>
                    {bot_msg}
                </div>
            </div>
            """
        
        if image_b64:
            html += f"""
            <div style='text-align: left; margin: 10px 0 20px 0;'>
                <img src="data:image/png;base64,{image_b64}" 
                     style="max-width: 384px; border-radius: 8px;" />
            </div>
            """
    
    return html

def load_selected_model(model_name):
    """Load the selected model."""
    global current_model
    try:
        logger.info(f"Requesting model load: {model_name}")
        
        # Provide user feedback
        status_msg = f"⏳ Loading {model_name}..."
        yield status_msg
        
        # Load the model
        model_loader.load_model(model_name)
        current_model = model_name
        
        final_msg = f"✅ Model loaded: {model_name}"
        logger.info(final_msg)
        return final_msg
        
    except FileNotFoundError as e:
        error_msg = f"❌ Model files not found: {str(e)}"
        logger.error(error_msg)
        return error_msg
    except RuntimeError as e:
        error_msg = f"❌ Failed to load model: {str(e)}"
        logger.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"❌ Unexpected error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return error_msg

    # Add this function after load_selected_model

def get_model_info():
    """Get current model information."""
    if model_loader.current_model_name:
        return f"Current: {model_loader.current_model_name}"
    else:
        return "No model loaded"

def new_chat(character):
    """Start a new chat."""
    global current_character, current_chat_id, chat_history
    
    current_character = character
    current_chat_id = chat_manager.generate_chat_id()
    chat_history = []
    
    # Add first message from character
    first_message = character_manager.get_first_message(character)
    if first_message:
        chat_history.append({"user": "", "bot": first_message})
    
    logger.info(f"Started new chat {current_chat_id} with {character}")
    return format_chat_html(chat_history), ""

def threaded_generation(user_input, character, model_type, temp, top_p, top_k, rep_penalty, max_tokens):
    """Handle response generation in a separate thread."""
    global chat_history
    
    with stream_lock:
        stream_state["text"] = ""
    done_event.clear()
    
    # Add user message to history
    chat_history.append({"user": user_input, "bot": ""})
    
    def run_generation():
        """Thread worker."""
        try:
            # Check for image generation trigger
            image_b64 = None
            if image_generator.should_generate_image(user_input):
                try:
                    logger.info("🎨 Generating image...")
                    image_b64 = image_generator.generate_image(user_input)
                except Exception as e:
                    logger.error(f"Image generation failed: {e}")
            
            # Generate text response
            logger.info("🤖 Generating response...")
            response = text_generator.generate_response(
                character, chat_history, model_type,
                temp, top_p, top_k, rep_penalty, max_tokens
            )
            
            # IMPORTANT: Only save the actual response, not the full output
            # The generator might return the prompt + response, so we need to clean it
            clean_response = response.strip()
            
            # Remove any echoed system prompts
            if clean_response.startswith("System:"):
                clean_response = clean_response.split("\n", 1)[-1].strip()
            
            # Remove character name if it's at the start
            if clean_response.startswith(f"{character}:"):
                clean_response = clean_response[len(f"{character}:"):].strip()
            
            logger.info(f"Clean response: {clean_response[:200]}...")
            
            # Simulate streaming (chunk by chunk)
            for i in range(0, len(clean_response), 30):
                with stream_lock:
                    stream_state["text"] = clean_response[:i + 30]
                time.sleep(0.05)
            
            # Update chat history with ONLY the clean response
            chat_history[-1]["bot"] = clean_response
            if image_b64:
                chat_history[-1]["image"] = image_b64
            
            # Save chat
            chat_manager.save_chat(character, current_chat_id, chat_history)
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            with stream_lock:
                stream_state["text"] = "I apologize, but I encountered an error."
            chat_history[-1]["bot"] = stream_state["text"]
        finally:
            done_event.set()
    
    # Start the thread
    threading.Thread(target=run_generation, daemon=True).start()

def handle_submit(user_input, character, model_type, temp, top_p, top_k, rep_penalty, max_tokens, stream_enabled):
    """Handle user message submission."""
    if not user_input.strip():
        return format_chat_html(chat_history), ""
    
    # Start generation thread
    threaded_generation(user_input, character, model_type, temp, top_p, top_k, rep_penalty, max_tokens)
    
    if stream_enabled:
        # Stream updates
        while not done_event.is_set():
            yield format_chat_html(chat_history), ""
            time.sleep(0.1)
    else:
        # Wait for completion
        done_event.wait()
    
    # Final update
    yield format_chat_html(chat_history), ""

def change_character(character):
    """Handle character selection change."""
    return new_chat(character)

# =========================
# UI Setup
# =========================

def create_ui():
    """Create the Gradio interface."""
    
    # Get available models
    available_models = get_available_models()
    if not available_models:
        logger.error("No models found in models/ directory!")
        available_models = [DEFAULT_MODEL]
    
    logger.info(f"Available models: {available_models}")
    
    with gr.Blocks(css="""
    #chat-area {
        height: calc(100vh - 280px);
        overflow-y: auto;
        padding: 10px;
        border: 1px solid #ccc;
        border-radius: 8px;
        background-color: #fafafa;
    }
    #input-container {
        display: flex;
        margin-top: 10px;
        width: 100%;
    }
    #textbox {
        flex-grow: 1;
        height: 60px !important;
        padding: 8px;
        margin: 0;
    }
    #send-button {
        width: 120px;
        margin-left: 8px;
        padding: 8px 16px;
    }
    #new-chat-button {
        width: 100%;
        margin-bottom: 10px;
    }
    """) as demo:
        
        gr.Markdown("# 🤖 Local AI Chatbot (Janitor.AI Clone)")
        gr.Markdown("*With conversation memory and image generation*")
        
        with gr.Row():
            with gr.Column(scale=3):
                chat_area = gr.HTML(label="Conversation", elem_id="chat-area")
                
                with gr.Row(elem_id="input-container"):
                    input_box = gr.Textbox(
                        lines=2,
                        placeholder="Type your message...",
                        elem_id="textbox",
                        show_label=False
                    )
                    submit_btn = gr.Button("Send", elem_id="send-button", variant="primary")
            
            # Replace the UI section starting from "# Model selector" with this:

            with gr.Column(scale=1):
                # Model selector
                model_dropdown = gr.Dropdown(
                    available_models,
                    label="🧠 Select Model",
                    value=available_models[0] if available_models else None
                )
                model_status = gr.Textbox(label="Model Status", value="No model loaded", interactive=False)
                current_model_display = gr.Textbox(
                    label="Current Model",
                    value="None",
                    interactive=False
                )
                load_model_btn = gr.Button("Load Model", variant="secondary")
    
                character_dropdown = gr.Dropdown(
                    character_manager.get_character_names(),
                    label="Select Character",
                    value=character_manager.get_character_names()[0] if character_manager.get_character_names() else None
                )
    
                new_chat_btn = gr.Button("🆕 New Chat", elem_id="new-chat-button")
    
                model_type_dropdown = gr.Dropdown(
                    ["llama2", "chatml", "alpaca", "vicuna", "simple"],
                    label="Prompt Format",
                    value="llama2",
                    info="Match your model type"
                )
    
                stream_checkbox = gr.Checkbox(label="Stream Response", value=True)
    
                with gr.Accordion("⚙️ Generation Settings", open=False):
                    max_tokens_slider = gr.Slider(32, 1024, value=DEFAULT_MAX_TOKENS, step=8, label="Max Tokens")
                    temp_slider = gr.Slider(0.1, 1.5, value=DEFAULT_TEMPERATURE, step=0.05, label="Temperature")
                    top_p_slider = gr.Slider(0.1, 1.0, value=DEFAULT_TOP_P, step=0.05, label="Top P")
                    top_k_slider = gr.Slider(0, 100, value=DEFAULT_TOP_K, step=1, label="Top K")
                    rep_penalty_slider = gr.Slider(1.0, 2.0, value=DEFAULT_REP_PENALTY, step=0.01, label="Repetition Penalty")

            # Event handlers
            load_model_btn.click(
                load_selected_model,
                inputs=[model_dropdown],
                outputs=[model_status]
            ).then(
                lambda: model_loader.current_model_name or "None",
                outputs=[current_model_display]
            )
        
        new_chat_btn.click(
            new_chat,
            inputs=[character_dropdown],
            outputs=[chat_area, input_box]
        )
        
        character_dropdown.change(
            change_character,
            inputs=[character_dropdown],
            outputs=[chat_area, input_box]
        )
        
        submit_btn.click(
            handle_submit,
            inputs=[
                input_box, character_dropdown, model_type_dropdown,
                temp_slider, top_p_slider, top_k_slider,
                rep_penalty_slider, max_tokens_slider, stream_checkbox
            ],
            outputs=[chat_area, input_box]
        )
        
        input_box.submit(
            handle_submit,
            inputs=[
                input_box, character_dropdown, model_type_dropdown,
                temp_slider, top_p_slider, top_k_slider,
                rep_penalty_slider, max_tokens_slider, stream_checkbox
            ],
            outputs=[chat_area, input_box]
        )
        
        # Load default model on startup
        demo.load(
            lambda: load_selected_model(available_models[0]) if available_models else "No models found",
            outputs=[model_status]
        )
        
        # Initialize first chat
        demo.load(
            new_chat,
            inputs=[character_dropdown],
            outputs=[chat_area, input_box]
        )
    
    return demo

# =========================
# Launch
# =========================

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(share=False, inbrowser=True)