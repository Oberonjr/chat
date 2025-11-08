import torch
from exllamav2 import ExLlamaV2, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2BaseGenerator
from exllamav2.generator.sampler import ExLlamaV2Sampler  # Import the sampler for Settings
from exllamav2.config import ExLlamaV2Config
from exllamav2.cache import ExLlamaV2Cache  # Import the cache class

# Update this to your actual model path
model_dir = "C:/Users/matei/llama-chatbot/models/Llama-2-7B-Chat_GPTQ"

# Load config
config = ExLlamaV2Config(model_dir)
model = ExLlamaV2(config)
model.load()
tokenizer = ExLlamaV2Tokenizer(config)

# Create cache for the model
cache = ExLlamaV2Cache(model, max_seq_len=2048, batch_size=1)  # Adjust max_seq_len and batch_size as needed

# Initialize generator with cache
generator = ExLlamaV2BaseGenerator(model, cache, tokenizer)

# Create and configure settings
settings = ExLlamaV2Sampler.Settings()
settings.token_repetition_penalty = 1.1
settings.temperature = 0.8
settings.top_p = 0.9
settings.top_k = 50

# Run inference
prompt = "<s>[INST] You are a helpful AI assistant. What is the capital of France? [/INST]"
output_text = generator.generate_simple(prompt, settings, 50)
# Print output
print(f"\nAssistant:{output_text}")


