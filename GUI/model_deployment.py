import gradio as gr
import torch
import tkinter as tk
import os
import uuid
from tkinter import filedialog
from PIL import Image

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

DEFAULT_OUTPUT_DIR = os.path.join(os.getcwd(), "outputs")

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
    all_images = []

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
        
        all_images.extend(output.images)

    # Save images to disk and return file paths
    saved_paths = []
    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    for img in all_images:
        filename = f"output_{uuid.uuid4().hex[:8]}.png"
        path = os.path.join(DEFAULT_OUTPUT_DIR, filename)
        img.save(path)
        saved_paths.append(path)

    return saved_paths

    