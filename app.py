import gradio as gr
import requests
import os

MODEL_ID = "ziadabdullah/saudi-dialect-translator"
API_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL_ID}"

# Optional: Add your HF token in Render Environment Variables as HF_TOKEN
HF_TOKEN = os.environ.get("HF_TOKEN", "")

def translate(text):
    if not text.strip():
        return ""
    
    payload = {"inputs": text}
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Content-Type": "application/json"
    }
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    
    try:
        response = requests.post(API_URL, json=payload, headers=headers, timeout=30)
        
        # Check if HF returned non-200 status
        if response.status_code != 200:
            return f"HF API Status {response.status_code}: {response.text}"
            
        try:
            result = response.json()
        except Exception:
            return f"Non-JSON response received: {response.text[:200]}"
            
        # Handle model cold start / loading state
        if isinstance(result, dict) and "error" in result:
            if "loading" in str(result["error"]).lower():
                est = result.get("estimated_time", 20)
                return f"⏳ Model is warming up on HF (~{int(est)}s). Please click Translate again in a few seconds!"
            return f"API Error: {result['error']}"
            
        # Parse standard seq2seq translation output
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
