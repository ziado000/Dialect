import gradio as gr
from huggingface_hub import InferenceClient
import os
import sys
import traceback

# Filter out harmless system errors from logs
class StderrFilter:
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr

    def write(self, s):
        if "Invalid file descriptor" in s or "BaseEventLoop.__del__" in s:
            return
        self.original_stderr.write(s)

    def flush(self):
        self.original_stderr.flush()

sys.stderr = StderrFilter(sys.stderr)

# Initialize HF Serverless Inference Client
model_id = "ziadabdullah/saudi-dialect-translator"
print(f"⏳ Connecting to Hugging Face Inference API for model: {model_id}")
client = InferenceClient(model=model_id)

def translate(text):
    if not text.strip():
        return ""
    
    try:
        # Try generic text_generation endpoint
        response = client.text_generation(
            text, 
            max_new_tokens=128,
            return_full_text=False
        )
        return response
    except Exception as e1:
        try:
            # Fallback to translation task endpoint if text_generation fails
            result = client.translation(text)
            if isinstance(result, dict) and "translation_text" in result:
                return result["translation_text"]
            return str(result)
        except Exception as e2:
            # Print complete error logs to UI and stdout for debugging
            err_msg = f"Primary Error: {str(e1)}\nFallback Error: {str(e2)}\n\nTraceback:\n{traceback.format_exc()}"
            print(f"❌ Translation Error: {err_msg}")
            return f"Error Details:\n{err_msg}"

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

# Launch with environment port binding for Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"🚀 Launching lightweight Gradio app on port {port}...")
    demo.launch(server_name="0.0.0.0", server_port=port)
