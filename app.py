import os
import gradio as gr
from huggingface_hub import InferenceClient

# Uses Hugging Face's free serverless inference API
client = InferenceClient(model="ziadabdullah/saudi-dialect-translator")

def translate(text):
    if not text.strip():
        return ""
    try:
        response = client.text_generation(text, max_new_tokens=128)
        return response
    except Exception as e:
        return f"Error: {str(e)}"

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
    print(f"Launching lightweight demo on port {port}...")
    demo.launch(server_name="0.0.0.0", server_port=port)
