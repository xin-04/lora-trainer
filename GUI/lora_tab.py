import gradio as gr
import subprocess
import os
import tkinter as tk
from tkinter import filedialog
import toml
from generate_lora_config import save_config

# Function to set all components to interactive
def set_all_interactive(components):
    for component in components:
        if isinstance(component, gr.Textbox) or isinstance(component, gr.Checkbox) or isinstance(component, gr.Dropdown):
            component.interactive = True
    
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

def lora_training_tab():
    with gr.Tab("LoRA Training Parameters"):

        gr.Markdown("## ✅ General Parameters")

        with gr.Group():
            gr.Markdown("### 📂 Paths")
            with gr.Row():
                pretrained_model_name_or_path = gr.Textbox(label="Base Model Path", placeholder="Path to base model", scale=4)
                browse_model_btn = gr.Button("📁", variant="secondary", scale=1)

            with gr.Row():
                vae_path = gr.Textbox(label="VAE Path", placeholder="Path to VAE", scale=4)
                browse_vae_btn = gr.Button("📁", variant="secondary", scale=1)    

            with gr.Row():
                train_data_dir = gr.Textbox(label="Training Dataset Folder", placeholder="Path to training dataset", scale=4)
                browse_data_btn = gr.Button("📁", variant="secondary", scale=1)

            with gr.Row():
                output_dir = gr.Textbox(label="LoRA Output Folder", placeholder="Path to training dataset", scale=4)
                output_dir_btn = gr.Button("📁", variant="secondary", scale=1)

            gr.Markdown("### 🔧 Network Settings")
            with gr.Row():
                resolution = gr.Textbox(label="Resolution", placeholder="512,512")
                bucket_reso_steps = gr.Textbox(label="Bucket Resolution Steps", placeholder="64")

            with gr.Row():
                network_dim = gr.Textbox(label="Network Dim", placeholder="128")
                network_alpha = gr.Textbox(label="Network Alpha", placeholder="64")
                seed = gr.Textbox(label="Seed", placeholder="1")

            gr.Markdown("### ⚙️ Training Settings")
            with gr.Row():
                learning_rate = gr.Textbox(label="Learning Rate", placeholder="5e-5")
                train_batch_size = gr.Textbox(label="Train Batch Size", placeholder="1")

            with gr.Row():
                max_train_steps = gr.Textbox(label="Max Training Steps", placeholder=2000)
                max_train_epochs = gr.Textbox(label="Max Train Epochs", placeholder=8)
            
            with gr.Row():
                output_name = gr.Textbox(label="Output Name", placeholder="lora_name")
                save_model_as = gr.Dropdown(
                    choices=["none", "ckpt", "pt", "safetensors"],
                    label="Save LoRA Format As:",
                    allow_custom_value=False
                )         
                

        gr.Markdown("## ⚙️ Advanced Parameters")

        with gr.Accordion("Show Advanced Parameters", open=False):
            with gr.Group():

                gr.Markdown("### 🔽 Precision Settings")

                with gr.Row():
                    save_precision = gr.Dropdown(
                        choices=["None", "float", "fp16", "bf16"],
                        label="Save Precision",
                        allow_custom_value=False
                    )
                    mixed_precision = gr.Dropdown(
                        choices=["no", "fp16", "bf16"],
                        label="Mixed Precision",
                        allow_custom_value=False
                    )                

                # General Settings for Advanced Parameters
                gr.Markdown("### 🛠️ General Settings (for advanced parameters)")
                with gr.Row():
                    huber_schedule = gr.Dropdown(
                        choices=["constant", "exponential", "snr"],
                        label="Scheduling Method for Huber Loss",
                        allow_custom_value=False
                    )
                    huber_c = gr.Number(label="Huber C")

                with gr.Row():
                    adaptive_noise_scale = gr.Number(label="Adaptive Noise Scale")
                    prior_loss_weight = gr.Number(label="Prior Loss Weight")
                    loss_type = gr.Dropdown(
                        choices=["l1", "l2", "huber", "smooth_l1"],
                        label="Loss Type",
                        allow_custom_value=False
                    )

                with gr.Row():
                    vae_batch_size = gr.Number(label="VAE Batch Size")
                    max_data_loader_n_workers = gr.Number(label="Max DataLoader Workers")
                    persistent_data_loader_workers = gr.Checkbox(label="Persistent Data Loader Workers")


                # Learning Rate Settings
                gr.Markdown("### ⚙️ Learning Rate Settings")
                with gr.Row():
                    unet_lr = gr.Number(label="UNet LR")
                    text_encoder_lr = gr.Number(label="Text Encoder LR")

                with gr.Row():
                    lr_scheduler = gr.Dropdown(
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup", "adafactor"],
                        label = "Learning Rate Scheduler",
                        allow_custom_value=False
                    )
                    lr_scheduler_num_cycles = gr.Number(label="LR Scheduler Cycles")
                    lr_scheduler_power = gr.Number(label="LR Scheduler Power")

                # Training Schedule
                gr.Markdown("### 📅 Training Schedule")
                with gr.Row():
                    max_token_length = gr.Dropdown(
                        choices=["None", "150", "225"],
                        label="Max Token Length",
                        allow_custom_value=False
                    )
                    max_timestep = gr.Number(label="Max Timestep")

                with gr.Row():
                    sample_sampler = gr.Dropdown(
                        choices=["ddim", "pndm", "lms", "euler", "huen"],
                        label="Sampler type for sample imgs",
                        allow_custom_value=False
                    )
                    sample_every_n_epochs = gr.Number(label="Sample Every N Epochs")

                with gr.Row():
                    save_every_n_steps = gr.Number(label="Save Every N Steps")
                    save_every_n_epochs = gr.Number(label="Save Every N Epochs")

                with gr.Row():
                    gradient_accumulation_steps = gr.Number(label="Gradient Accumulation Steps")
                    clip_skip = gr.Number(label="Clip Skip")

                # Noise & Optimization Settings
                gr.Markdown("### 🔊 Noise & Optimization Settings")
                with gr.Row():
                    multires_noise_discount = gr.Number(label="Multires Noise Discount")
                    min_snr_gamma = gr.Number(label="Min SNR Gamma")
                    noise_offset = gr.Number(label="Noise Offset")

                gr.Markdown("### 🏷️ Memory Optimization")
                with gr.Row():
                    highvram = gr.Checkbox(label="High VRAM")
                    lowram = gr.Checkbox(label="Low RAM")
                    xformers = gr.Textbox(label="Enable Xformers", placeholder="xformers")
                    sdpa = gr.Checkbox(label="Enable SDPA")

                gr.Markdown("### 🔄 Caching & Upscaling")
                with gr.Row():
                    bucket_no_upscale = gr.Checkbox(label="Bucket No Upscale")
                    cache_latents = gr.Checkbox(label="Cache Latents")
                    cache_latents_to_disk = gr.Checkbox(label="Cache Latents to Disk")

                gr.Markdown("### 🧰 Optimizer & Network Flags")
                with gr.Row():
                    optimizer_type = gr.Dropdown(
                        choices=["adafactor", "adamw8bit", "adamw"],
                        label="Optimizer Type",
                        allow_custom_value=False
                    )
                    optimizer_args = gr.Textbox(label="Optimizer Arguments")
                
                with gr.Row():
                    use_8bit_adam = gr.Checkbox(label="Use 8-bit Adam")
                    use_lion_optimizer = gr.Checkbox(label="Use Lion Optimizer")
                
                with gr.Row():
                    network_train_unet_only = gr.Checkbox(label="Train UNet Only")
                    network_train_text_encoder_only = gr.Checkbox(label="Train Text Encoder Only")
                    dim_from_weights = gr.Checkbox(label="DIM from Weights")

                gr.Markdown("### ⚙️ Performance Flags")
                with gr.Row():
                    no_half_vae = gr.Checkbox(label="No Half VAE")
                    fp8_base = gr.Checkbox(label="FP8 Base")
                    full_bf16 = gr.Checkbox(label="Full BF16")
                    full_fp16 = gr.Checkbox(label="Full FP16")

                gr.Markdown("### ⚡ Checkpointing & Caching Flags")
                with gr.Row():
                    gradient_checkpointing = gr.Checkbox(label="Gradient Checkpointing")
                    skip_cache_check = gr.Checkbox(label="Skip Cache Check")
                    enable_bucket = gr.Checkbox(label="Enable Bucket")

        gr.Markdown('## 📝 Config Management')
        with gr.Row():
            save_btn = gr.Button("💾 Save Config")
            load_btn = gr.Button("📂 Load Config")

        gr.Markdown('## ⌛ Time to Train!')
        with gr.Row():
                config_path = gr.Textbox(label="Config Path", placeholder="Path to config", scale=4)
                browse_config_btn = gr.Button("📁", variant="secondary", scale=1)
        
        train_btn = gr.Button("Start Training")

        gr.Markdown("## 📄 Your Status")
        status_output = gr.Textbox(label="Status", placeholder="Waiting for user action...")

        def flatten_dict(d, parent_key='', sep='.'):
            items = {}
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.update(flatten_dict(v, new_key, sep=sep))
                else:
                    items[new_key] = v
            return items


        def on_load_click():
            # Load the configuration from the TOML file
            config = load_config()
            config = flatten_dict(config)
            
            gradio_components = {
                "pretrained_model_name_or_path": pretrained_model_name_or_path,
                "vae_path": vae_path,
                "train_data_dir": train_data_dir,
                "output_dir": output_dir,
                "resolution": resolution,
                "bucket_reso_steps": bucket_reso_steps,
                "network_dim": network_dim,
                "network_alpha": network_alpha,
                "seed": seed,
                "learning_rate": learning_rate,
                "train_batch_size": train_batch_size,
                "max_train_steps": max_train_steps,
                "max_train_epochs": max_train_epochs,
                "output_name": output_name,
                "save_model_as": save_model_as,
                "save_precision": save_precision,
                "mixed_precision": mixed_precision,
                "huber_schedule": huber_schedule,
                "huber_c": huber_c,
                "adaptive_noise_scale": adaptive_noise_scale,
                "prior_loss_weight": prior_loss_weight,
                "loss_type": loss_type,
                "vae_batch_size": vae_batch_size,
                "max_data_loader_n_workers": max_data_loader_n_workers,
                "persistent_data_loader_workers": persistent_data_loader_workers,
                "unet_lr": unet_lr,
                "text_encoder_lr": text_encoder_lr,
                "lr_scheduler": lr_scheduler,
                "lr_scheduler_num_cycles": lr_scheduler_num_cycles,
                "lr_scheduler_power": lr_scheduler_power,
                "max_token_length": max_token_length,
                "max_timestep": max_timestep,
                "sample_sampler": sample_sampler,
                "sample_every_n_epochs": sample_every_n_epochs,
                "save_every_n_steps": save_every_n_steps,
                "save_every_n_epochs": save_every_n_epochs,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "clip_skip": clip_skip,
                "multires_noise_discount": multires_noise_discount,
                "min_snr_gamma": min_snr_gamma,
                "noise_offset": noise_offset,
                "highvram": highvram,
                "lowram": lowram,
                "xformers": xformers,
                "sdpa": sdpa,
                "bucket_no_upscale": bucket_no_upscale,
                "cache_latents": cache_latents,
                "cache_latents_to_disk": cache_latents_to_disk,
                "optimizer_type": optimizer_type,
                "optimizer_args": optimizer_args,
                "use_8bit_adam": use_8bit_adam,
                "use_lion_optimizer": use_lion_optimizer,
                "network_train_unet_only": network_train_unet_only,
                "network_train_text_encoder_only": network_train_text_encoder_only,
                "dim_from_weights": dim_from_weights,
                "no_half_vae": no_half_vae,
                "fp8_base": fp8_base,
                "full_bf16": full_bf16,
                "full_fp16": full_fp16,
                "gradient_checkpointing": gradient_checkpointing,
                "skip_cache_check": skip_cache_check,
                "enable_bucket": enable_bucket
            }


            updates = []
            for comp in gradio_components:
                key = component_mapping.get(comp, comp)
                value = config.get(key, None)

                if isinstance(comp[0], gr.Checkbox):
                    updates.append(gr.update(value=bool(value)))
                elif isinstance(comp[0], gr.Textbox):
                    updates.append(gr.update(value=str(value) if value is not None else ""))
                elif isinstance(comp[0], gr.Number):
                    try:
                        updates.append(gr.update(value=float(value)))
                    except (TypeError, ValueError):
                        updates.append(gr.update(value=None))
                elif isinstance(comp[0], gr.Dropdown):
                    updates.append(gr.update(value=value))
                else:
                    updates.append(gr.update(value=value))

            return updates


        browse_model_btn.click(browse_pretrained_model_name_or_path, inputs=[], outputs=pretrained_model_name_or_path)
        browse_vae_btn.click(browse_vae_path, inputs=[], outputs=vae_path)
        browse_data_btn.click(browse_training_data_dir, inputs=[], outputs=train_data_dir)
        output_dir_btn.click(browse_output_dir, inputs=[], outputs=output_dir)
        browse_config_btn.click(browse_config_path, inputs=[], outputs=config_path)

        all_components = [
            pretrained_model_name_or_path, 
            vae_path,
            train_data_dir,
            output_dir,
            resolution, 
            bucket_reso_steps,
            network_dim, 
            network_alpha, 
            seed,
            learning_rate, 
            train_batch_size,
            max_train_steps, 
            max_train_epochs,
            output_name, 
            save_model_as,
            save_precision, 
            mixed_precision,
            huber_schedule, 
            huber_c,
            adaptive_noise_scale, 
            prior_loss_weight, 
            loss_type,
            vae_batch_size, 
            max_data_loader_n_workers,
            persistent_data_loader_workers,
            unet_lr, 
            text_encoder_lr,
            lr_scheduler, 
            lr_scheduler_num_cycles, 
            lr_scheduler_power,
            max_token_length, 
            max_timestep,
            sample_sampler, 
            sample_every_n_epochs,
            save_every_n_steps, 
            save_every_n_epochs,
            gradient_accumulation_steps, 
            clip_skip,
            multires_noise_discount, 
            min_snr_gamma, 
            noise_offset,
            highvram, 
            lowram, 
            xformers, 
            sdpa,
            bucket_no_upscale, 
            cache_latents, 
            cache_latents_to_disk,
            optimizer_type, 
            optimizer_args,
            use_8bit_adam, 
            use_lion_optimizer,
            network_train_unet_only, 
            network_train_text_encoder_only, 
            dim_from_weights,
            no_half_vae, 
            fp8_base, 
            full_bf16, 
            full_fp16,
            gradient_checkpointing, 
            skip_cache_check, 
            enable_bucket]

        set_all_interactive(all_components)

        save_btn.click(
            fn=save_config,
            inputs=[
                pretrained_model_name_or_path, vae_path, train_data_dir, output_dir,
                resolution, bucket_reso_steps, network_dim, network_alpha, seed,
                learning_rate, train_batch_size, max_train_steps, max_train_epochs,
                output_name, save_model_as, save_precision, mixed_precision,
                huber_schedule, huber_c, adaptive_noise_scale, prior_loss_weight, loss_type,
                vae_batch_size, max_data_loader_n_workers, persistent_data_loader_workers,
                unet_lr, text_encoder_lr, lr_scheduler, lr_scheduler_num_cycles, lr_scheduler_power,
                max_token_length, max_timestep, sample_sampler, sample_every_n_epochs,
                save_every_n_steps, save_every_n_epochs, gradient_accumulation_steps, clip_skip,
                multires_noise_discount, min_snr_gamma, noise_offset, highvram, lowram, xformers, 
                sdpa, bucket_no_upscale, cache_latents, cache_latents_to_disk,
                optimizer_type, optimizer_args, use_8bit_adam, use_lion_optimizer,
                network_train_unet_only, network_train_text_encoder_only, dim_from_weights,
                no_half_vae, fp8_base, full_bf16, full_fp16,
                gradient_checkpointing, skip_cache_check, enable_bucket
            ],
            outputs=[status_output]
        )

        load_btn.click(
            fn=on_load_click,
            inputs=[],
            outputs=all_components  # return must match order of these components
        )

        def run_training(config_path):
            try:
                # Construct the command to call the batch file with the config file path
                command = f'start cmd.exe /K "C:\\lora-trainer\\train.bat {config_path}"'
                
                # Run the command in a subprocess with real-time output capture
                process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                # Stream the output to the output textbox in real-time
                while True:
                    if process.poll() is not None:  # Check if the process has finished
                        break

            except Exception as e:
                return f"An error occurred: {str(e)}"


        train_btn.click(
            fn=run_training,
            inputs=[config_path],
            outputs=[]  # Update the same output textbox with process output
        )

        
