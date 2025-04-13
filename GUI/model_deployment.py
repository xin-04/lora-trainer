import gradio as gr
import torch
import time
import os
from diffusers import (
    StableDiffusionPipeline,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    DDIMScheduler,
    PNDMScheduler,
    DPMSolverMultistepScheduler,
    LMSDiscreteScheduler,
    HeunDiscreteScheduler,
)
from safetensors.torch import load_file
import tkinter as tk
from tkinter import filedialog

SCHEDULER_MAPPING = {
    "Euler A (euler_a)": EulerAncestralDiscreteScheduler,
    "Euler": EulerDiscreteScheduler,
    "DDIM": DDIMScheduler,
    "PLMS": PNDMScheduler,
    "DPM++": DPMSolverMultistepScheduler,
    "LMS": LMSDiscreteScheduler,
    "Heun": HeunDiscreteScheduler,
}


def load_lora():
    default_dir = os.path.abspath("C:\lora-trainer\models\lora")
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        initialdir = default_dir,
        title="Select a LoRA",
        defaultextension=".safetensors",
        filetypes=[("SAFETENSORS files", "*.safetensors")]
    )

    return file_path

# Load once to avoid repeated model initialization
def load_pipeline_lora(lora_path, lora_scale, scheduler_name):
    base_model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    lora_path = lora_path

    scheduler_cls = SCHEDULER_MAPPING.get(scheduler_name, EulerAncestralDiscreteScheduler)
    scheduler = scheduler_cls.from_pretrained(base_model_id, subfolder="scheduler")

    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_id,
        scheduler=scheduler,
        torch_dtype=torch.float16,
    ).to("cuda")

    pipe.load_lora_weights(lora_path, weight_name="mads.safetensors", scale=lora_scale)
    return pipe

def load_pipeline(scheduler_name):
    base_model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"

    scheduler_cls = SCHEDULER_MAPPING.get(scheduler_name, EulerAncestralDiscreteScheduler)
    scheduler = scheduler_cls.from_pretrained(base_model_id, subfolder="scheduler")

    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_id,
        scheduler=scheduler,
        torch_dtype=torch.float16,
    ).to("cuda")

    return pipe

# Gradio progress-integrated image generation
def generate_image(scheduler_name, lora_path, lora_scale, width, height, 
                   prompt, negative_prompt, steps, guidance, 
                   seed, batch_count, batch_size, 
                   progress=gr.Progress(track_tqdm=True)):
    
    seed = int(seed)
    width = int(width)
    height = int(height)
    
    if lora_path.strip() == "":
        pipe = load_pipeline(scheduler_name)
    else:
        pipe = load_pipeline_lora(lora_path, lora_scale, scheduler_name)
    
    if seed is None or seed == -1:
        seed = int(torch.seed())  # random seed
    generator = torch.manual_seed(seed)

    total = batch_count * batch_size
    images = []

    with torch.autocast("cuda"):
        # Simulate step-wise progress (optional)
        for i in range(batch_count):
            progress(i, total=batch_count)
        
        output = pipe(
            [prompt] * batch_size,
            negative_prompt=[negative_prompt] * batch_size,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=width,
            height=height,
            generator=generator
        )
        
        images.extend(output.images)

    return images

# Gradio tab for deployment
def model_deployment_tab():
    with gr.Tab("Model Deployment"):

        prompt = gr.TextArea(label="Prompt", value="modelshoot style, serious expression, full face, handsome face, high detail, intricate, sharp focus, photorealistic, caucasian, middle-aged man, black shirt, black necktie")
        negative_prompt = gr.TextArea(label="Negative Prompt", value="deformed, distorted, disfigures, blurry, low-res, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands and fingers, disconeected limbs, mutation, mutated, ugly, disgusting, amputation")
        
        with gr.Row():
            lora_path = gr.Textbox(label="LoRA Path", placeholder="Path to LoRA", scale=4)
            browse_lora_btn = gr.Button("📁", variant="secondary", scale=1)    

        lora_scale = gr.Number(label="Enter influence of LoRA onto the base model (0.0 - 1.0)")

        with gr.Row():
            width = gr.Number(label="Image Width:")
            height = gr.Number(label="Image Height:")

        with gr.Row():
            scheduler_name = gr.Dropdown(
                label="Sampler (Scheduler)",
                choices=list(SCHEDULER_MAPPING.keys()),
                value="Euler A (euler_a)"
            )

        with gr.Row():
            steps = gr.Slider(10, 100, value=50, step=1, label="Steps")
            guidance = gr.Slider(1.0, 15.0, value=7.5, step=0.1, label="Guidance Scale")
            seed = gr.Number(value=-1, label="Seed (-1 = random)")

        with gr.Row():
            batch_count = gr.Slider(1, 10, value=1, step=1, label="Batch Count")
            batch_size = gr.Slider(1, 8, value=1, step=1, label="Batch Size")

        generate_btn = gr.Button("Generate")
        output_gallery = gr.Gallery(label="Generated Images").style(grid=[2], height="auto")

        browse_lora_btn.click(fn=load_lora, inputs=[], outputs=lora_path)
        generate_btn.click(
            fn=generate_image,
            inputs=[scheduler_name, lora_path, lora_scale, width, height, 
                    prompt, negative_prompt, steps, guidance, 
                    seed, batch_count, batch_size],
            outputs=output_gallery
        )
