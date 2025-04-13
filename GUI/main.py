import gradio as gr
import webbrowser
import threading
import time

from lora_tab import lora_training_tab
from dataset_preparation import dataset_preparation_tab
from model_deployment import model_deployment_tab

# Placeholder functions
def prepare_dataset(data_dir):
    return f"Dataset prepared from: {data_dir}"

def deploy_model(model_path):
    return f"Deployed model from: {model_path}"


def create_interface():
    # Gradio Interface
    with gr.Blocks(title="LoRA Training GUI") as demo:
        dataset_preparation_tab()

        lora_training_tab()

        model_deployment_tab()

    return demo

# Open browser before launching the app
def launch_with_browser():
    url = "http://127.0.0.1:7861"

    # Start browser opener in background thread
    def open_browser():
        time.sleep(2)  # Wait briefly for server to start
        webbrowser.open(url)

    threading.Thread(target=open_browser, daemon=True).start()

    # Launch Gradio (blocking)
    app = create_interface()
    app.queue()
    app.launch(server_name="127.0.0.1", server_port=7861, share=False, inbrowser=False, quiet=True, prevent_thread_lock=False)

    
if __name__ == "__main__":
    launch_with_browser()