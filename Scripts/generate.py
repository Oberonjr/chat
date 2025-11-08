# import gradio as gr
# import torch
# import json
# import requests
# from auto_gptq import AutoGPTQForCausalLM, AutoGPTQTokenizer

# # ==== Load characters ====
# with open("app/character_config.json", "r", encoding="utf-8") as f:
#     characters = json.load(f)

# character_names = list(characters.keys())

# # ==== Load LLaMA model ====
# model_path = "C:/Users/matei/llama-chatbot/models/Llama-2-7B-Chat-GPTQ"
# tokenizer = AutoGPTQTokenizer.from_pretrained(
#     model_path,
#     use_fast=True,
#     local_files_only=True  # ✅ Force local-only loading
# )

# model = AutoGPTQForCausalLM.from_quantized(
#     model_path,
#     device="cuda",
#     use_safetensors=True,
#     trust_remote_code=True,
#     local_files_only=True  # ✅ Also force this for safety
# )

# print("Using GPTQ tokenizer:", AutoGPTQTokenizer)

# # ==== Generate text ====
# def generate_personality_response(character_name, input_text):
#     personality_prompt = characters[character_name]["personality_prompt"]
#     full_prompt = f"{personality_prompt}\nUser: {input_text}\n{character_name}:"
#     inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
#     with torch.no_grad():
#         outputs = model.generate(inputs['input_ids'], max_new_tokens=200)
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

# # ==== Generate image via A1111 ====
# def generate_image(prompt):
#     payload = {
#         "prompt": prompt,
#         "steps": 20,
#         "cfg_scale": 7,
#         "width": 512,
#         "height": 512
#     }
#     response = requests.post("http://127.0.0.1:7860/sdapi/v1/txt2img", json=payload)
#     r = response.json()
#     image_data = r['images'][0]

#     # Decode base64 to image for Gradio
#     import base64
#     from io import BytesIO
#     from PIL import Image
#     image = Image.open(BytesIO(base64.b64decode(image_data.split(",", 1)[0])))
#     return image

# # ==== Main chatbot handler ====
# def chatbot_response(input_text, character_name):
#     trigger_words = ["show me", "picture of", "draw", "image of"]
#     if any(word in input_text.lower() for word in trigger_words):
#         return None, generate_image(input_text)
#     else:
#         return generate_personality_response(character_name, input_text), None

# # ==== Gradio UI ====
# with gr.Blocks() as demo:
#     gr.Markdown("# 🤖 LLaMA Chatbot (Character + Image Gen)")

#     with gr.Row():
#         character = gr.Dropdown(character_names, label="Choose Character", value=character_names[0])

#     input_box = gr.Textbox(label="Say something...")
#     text_output = gr.Textbox(label="Character Response")
#     image_output = gr.Image(label="Generated Image")

#     submit = gr.Button("Send")

#     submit.click(fn=chatbot_response, inputs=[input_box, character], outputs=[text_output, image_output])

# demo.launch()


  

import gradio as gr
import torch
import json
import requests
import traceback
from auto_gptq import AutoGPTQForCausalLM, AutoGPTQTokenizer
print("[DEBUG] Starting chatbot...")

# ==== Load characters ====
try:
    print("[DEBUG] Loading character config...")
    with open("app/character_config.json", "r", encoding="utf-8") as f:
        characters = json.load(f)
    character_names = list(characters.keys())
    print("[DEBUG] Characters loaded:", character_names)
except Exception as e:
    print("[ERROR] Failed to load characters:", e)
    traceback.print_exc()

# ==== Load LLaMA model ====
model_path = "C:/Users/matei/llama-chatbot/models/Llama-2-7B-Chat-GPTQ"
import os
print("[DEBUG] Model directory contents:", os.listdir(model_path))
print("[DEBUG] Model path:", model_path)

# Tokenizer loading debug
try:
    print("[DEBUG] Loading tokenizer...")
    tokenizer = AutoGPTQTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        local_files_only=True
    )
    print("[DEBUG] Tokenizer loaded successfully.")
except Exception as e:
    print("[ERROR] Tokenizer loading failed!")
    print("[EXCEPTION]", type(e).__name__, str(e))
    traceback.print_exc()

# Model loading debug
try:
    print("[DEBUG] Loading quantized model...")
    model = AutoGPTQForCausalLM.from_quantized(
        model_path,
        device="cuda",
        use_safetensors=True,
        trust_remote_code=True,
        local_files_only=True
    )
    print("[DEBUG] Model loaded successfully.")
except Exception as e:
    print("[ERROR] Model loading failed!")
    print("[EXCEPTION]", type(e).__name__, str(e))
    traceback.print_exc()
