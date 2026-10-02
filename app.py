import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import asyncio
import os
import sys

# Filter out harmless system errors from logs
class StderrFilter:
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr

    def write(self, s):
        # Filter out the specific asyncio error
        if "Invalid file descriptor" in s or "BaseEventLoop.__del__" in s:
            return
        self.original_stderr.write(s)

    def flush(self):
        self.original_stderr.flush()

# Apply the filter
sys.stderr = StderrFilter(sys.stderr)

# Load the trained model
model_path = "ziadabdullah/saudi-dialect-translator"
print(f"⏳ Loading model from: {model_path}")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("⚠️ Make sure you unzipped the model correctly!")
    exit()

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

def translate(text):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    # Force Arabic output token
    forced_bos_token_id = tokenizer.convert_tokens_to_ids("arb_Arab")
    
    outputs = model.generate(
        **inputs,
        forced_bos_token_id=forced_bos_token_id,
        max_length=128,
        num_beams=5,
        early_stopping=True
    )
    
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translation

# Create the Web UI
with gr.Blocks() as demo:
    gr.Markdown(
        """
        # 🇸🇦 Saudi Dialect AI Translator
        ### Translate English to Authentic Saudi Dialect
        """
    )
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="English Text", 
                placeholder="Type here... (e.g., Hello, how are you?)",
                lines=3
            )
            translate_btn = gr.Button("Translate 🚀", variant="primary")
            
        with gr.Column():
            output_text = gr.Textbox(
                label="Saudi Dialect Translation", 
                lines=3,
                interactive=False
            )
            
    # Examples
    gr.Examples(
        examples=[
            ["Hello, how are you?"],
            ["What is your name?"],
            ["I am from Riyadh"],
            ["The food is delicious"],
            ["See you later"]
        ],
        inputs=input_text
    )
    
    translate_btn.click(fn=translate, inputs=input_text, outputs=output_text)

# Launch
print("Launching demo...")
demo.launch()
