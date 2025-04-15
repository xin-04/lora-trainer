import gradio as gr
import tkinter as tk
import os
import toml
from tkinter import filedialog

# Function to set all components to interactive
def set_all_interactive(components):
    for component in components:
        if isinstance(component, gr.Textbox) or isinstance(component, gr.Checkbox) or isinstance(component, gr.Dropdown):
            component.interactive = True

component_mapping = {
    # model
    "model_path": "model.pretained_model_name_or_path",
    "output_name": "model.output_name",
    "train_data_dir": "model.train_data_dir",
    "save_model_as": "model.save_model_as",

    # folders
    "output_dir": "folders.output_dir",

    # accelerate_launch
    "mixed_precision": "accelerate_launch.mixed_precision",

    # basic
    "cache_latents": "basic.cache_latents",
    "cache_latents_to_disk": "basic.cache_latents_to_disk",
    "enable_bucket": "basic.enable_bucket",
    "highvram": "basic.highvram",
    "lowram": "basic.lowram",
    "learning_rate": "basic.learning_rate",
    "lr_scheduler": "basic.lr_scheduler",
    "lr_scheduler_num_cycles": "basic.lr_scheduler_num_cycles",
    "lr_scheduler_power": "basic.lr_scheduler_power",
    "network_dim": "basic.network_dim",
    "network_alpha": "basic.network_alpha",
    "resolution": "basic.max_resolution",
    "max_train_steps": "basic.max_train_steps",
    "max_train_epochs": "basic.max_train_epochs",
    "bucket_reso_steps": "basic.min_bucket_reso",
    "optimizer_type": "basic.optimizer",
    "optimizer_args": "basic.optimizer_args",
    "use_8bit_adam": "basic.use_8bit_adam",
    "use_lion_optimizer": "basic.use_lion_optimizer",
    "save_every_n_epochs": "basic.save_every_n_epochs",
    "save_every_n_steps": "basic.save_every_n_steps",
    "save_precision": "basic.save_precision",
    "seed": "basic.seed",
    "train_batch_size": "basic.train_batch_size",
    "text_encoder_lr": "basic.text_encoder_lr",
    "unet_lr": "basic.unet_lr",

    # advanced
    "adaptive_noise_scale": "advanced.adaptive_noise_scale",
    "bucket_no_upscale": "advanced.bucket_no_upscale",
    "bucket_reso_steps": "advanced.bucket_reso_steps",
    "clip_skip": "advanced.clip_skip",
    "dim_from_weights": "advanced.dim_from_weights",
    "fp8_base": "advanced.fp8_base",
    "full_bf16": "advanced.full_bf16",
    "full_fp16": "advanced.full_fp16",
    "gradient_accumulation_steps": "advanced.gradient_accumulation_steps",
    "gradient_checkpointing": "advanced.gradient_checkpointing",
    "highvram": "advanced.highvram",
    "huber_c": "advanced.huber_c",
    "huber_schedule": "advanced.huber_schedule",
    "loss_type": "advanced.loss_type",
    "lowram": "advanced.lowram",
    "max_data_loader_n_workers": "advanced.max_data_loader_n_workers",
    "max_timestep": "advanced.max_timestep",
    "max_token_length": "advanced.max_token_length",
    "min_snr_gamma": "advanced.min_snr_gamma",
    "multires_noise_discount": "advanced.multires_noise_discount",
    "network_train_unet_only": "advanced.network_train_unet_only",
    "network_train_text_encoder_only": "advanced.network_train_text_encoder_only",
    "no_half_vae": "advanced.no_half_vae",
    "noise_offset": "advanced.noise_offset",
    "persistent_data_loader_workers": "advanced.persistent_data_loader_workers",
    "prior_loss_weight": "advanced.prior_loss_weight",
    "save_every_n_steps_advanced": "advanced.save_every_n_steps",  # separate name to prevent overwrite
    "skip_cache_check": "advanced.skip_cache_check",
    "vae_batch_size": "advanced.vae_batch_size",
    "vae_path": "advanced.vae",
    "xformers": "advanced.xformers",
    "sdpa": "advanced.sdpa",

    # dataset_preparation
    "images_folder": "dataset_preparation.images_folder",

    # samples
    "sample_every_n_epochs": "samples.sample_every_n_steps",  # reused var name from basic
    "sample_sampler": "samples.sample_sampler"
}

def load_config():
    default_dir = os.path.abspath("C:\lora-trainer\lora_presets")
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        initialdir = default_dir,
        title="Select a TOML file",
        defaultextension=".toml",
        filetypes=[("TOML files", "*.toml")]
    )

    with open(file_path, 'r') as f:
        config = toml.load(f)

    return config

def browse_config_path():
    default_dir = os.path.abspath("C:\lora-trainer\lora_presets")
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        initialdir = default_dir,
        title="Select a TOML file",
        defaultextension=".toml",
        filetypes=[("TOML files", "*.toml")]
    )

    return file_path

def browse_pretrained_model_name_or_path():
    root = tk.Tk()
    root.withdraw()
    default_path = os.path.abspath(r"C:\lora-trainer\models\unet")

    pretrained_model_name_or_path = filedialog.askopenfilename(
        initialdir = default_path,
        title = "Select Base Model File",
        filetypes = [("Model files", "*.*")]  
    )

    return pretrained_model_name_or_path or ""

def browse_vae_path():
    root = tk.Tk()
    root.withdraw()
    default_path = os.path.abspath(r"C:\lora-trainer\models\vae")

    vae_path = filedialog.askopenfilename(
        initialdir = default_path,
        title = "Select Base Model File",
        filetypes = [("Model files", "*.*")]  
    )

    return vae_path or ""

def browse_training_data_dir():
    root = tk.Tk()
    root.withdraw()
    default_path = os.path.abspath(r"C:\lora-trainer\dataset")

    training_data_dir = filedialog.askdirectory(
        initialdir = default_path,
        title = "Select Training Data Directory",
    )

    return training_data_dir or ""

def browse_output_dir():
    root = tk.Tk()
    root.withdraw()
    default_path = os.path.abspath(r"C:\lora-trainer\models\lora")

    output_path = filedialog.askdirectory(
        initialdir = default_path,
        title = "Select LoRA Output Directory",
    )

    return output_path or ""



        
