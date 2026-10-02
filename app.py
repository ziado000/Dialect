import gradio as gr
import requests
import os

MODEL_ID = "ziadabdullah/saudi-dialect-translator"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"

def translate(text):
    if not text.strip():
        return ""
    
    payload = {"inputs": text}
    
    try:
        response = requests.post(API_URL, json=payload, timeout=30)
        result = response.json()
        
        # If model is waking up (cold start)
        if isinstance(result, dict) and "error" in result:
            if "loading" in result["error"].lower():
                estimated_time = result.get("estimated_time", 20)
                return f"⏳ Model is warming up on HF servers (~{int(estimated_time)}s). Please click Translate again in a moment!"
            return f"API Error: {result['error']}"
            
        # Parse output for translation models
        if isinstance(result, list) and len(result) > 0:
            if "translation_text" in result[0]:
                return result[0]["translation_text"]
            elif "generated_text" in result[0]:
                return result[0]["generated_text"]
        
        return str(result)
        
    except Exception as e:
        return f"Request Exception: {str(e)}"

# UI Layout
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
