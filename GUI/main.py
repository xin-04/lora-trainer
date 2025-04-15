import gradio as gr
import webbrowser
import threading
import time
import subprocess

from generate_lora_config import save_config

from dataset_preparation import (
    save_uploaded_image,
    generate_captions_for_folder,
    load_dataset_dir,
    load_preedit_image,
    free_crop_image,
    crop_image,
    resize_image,
    resize_image_by_ratio
)

from lora_tab import (
    set_all_interactive,
    component_mapping,
    load_config,
    browse_config_path,
    browse_output_dir,
    browse_pretrained_model_name_or_path,
    browse_training_data_dir,
    browse_vae_path
)

from model_deployment import (
    SCHEDULER_MAPPING,
    load_lora,
    generate_image
)

def create_interface():
    # Gradio Interface
    with gr.Blocks(title="LoRA Training GUI") as demo:

        with gr.Tab("Dataset Preparation"):
            
            gr.Markdown("### ⬆️ Upload Pre-edited Image")
            with gr.Group():
                with gr.Row():
                    file_upload = gr.File(label="Upload Image", file_types=[".png", ".jpg", ".jpeg", ".webp"])
                    upload_status = gr.Textbox(label="Upload Status")
            
            gr.Markdown("### 👇 Select Image To Crop / Resize")
            with gr.Group():
                with gr.Row():
                    preedit_img_path = gr.Textbox(label="Select Image Path", placeholder="Path to pre-edited image", scale=4)
                    browse_preedit_img_btn = gr.Button("📁", variant="secondary", scale=1)
                
            gr.Markdown("### ✂️ Image Cropping Tool")
            with gr.Row():
                free_crop_btn = gr.Button("Crop Image Freely")            
                crop_btn = gr.Button("Crop Image with Aspect")

            gr.Markdown("### 🪟 Image Resizing Tool")  
            with gr.Group():
                with gr.Row():
                    width = gr.Number(label="Enter Width:")
                    height = gr.Number(label="Enter Height:")
                    resize_image_btn = gr.Button("Resize", variant="secondary")
                with gr.Row():
                    ratio = gr.Number(label="Enter Ratio")
                    resize_image_ratio_btn = gr.Button("Resize while keeping aspect", variant="secondary")

            with gr.Group():
                gr.Markdown("### 📄 Auto-captioning Tool")
                with gr.Row():
                    dataset_dir = gr.Textbox(label="Select Dataset Directory", placeholder="Path to dataset", scale=4)
                    dataset_dir_btn = gr.Button("📁", variant="secondary", scale=1)
                caption_btn = gr.Button("Auto-captioning")

            gr.Markdown("### ⌛ Your Current Status") 
            output_text = gr.Textbox(label="Output Status")

            dataset_dir_btn.click(fn=load_dataset_dir, inputs=[], outputs=dataset_dir)
            browse_preedit_img_btn.click(fn=load_preedit_image, inputs=[], outputs=preedit_img_path)
            free_crop_btn.click(fn=free_crop_image, inputs=[preedit_img_path], outputs=output_text)
            crop_btn.click(fn=crop_image, inputs=[preedit_img_path], outputs=output_text)
            file_upload.change(fn=save_uploaded_image, inputs=[file_upload], outputs=[upload_status])
            resize_image_btn.click(fn=resize_image, inputs=[preedit_img_path, width, height], outputs=output_text)
            resize_image_ratio_btn.click(fn=resize_image_by_ratio, inputs=[preedit_img_path, ratio], outputs=output_text)
            caption_btn.click(fn=generate_captions_for_folder, inputs=dataset_dir, outputs=[])


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

            with gr.Group():
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

        with gr.Tab("Model Deployment"):

            prompt = gr.TextArea(label="Prompt", value="modelshoot style, serious expression, full face, handsome face, high detail, intricate, sharp focus, photorealistic, caucasian, middle-aged man, black shirt, black necktie")
            negative_prompt = gr.TextArea(label="Negative Prompt", value="deformed, distorted, disfigures, blurry, low-res, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands and fingers, disconeected limbs, mutation, mutated, ugly, disgusting, amputation")
            
            with gr.Group():
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
            output_gallery = gr.Gallery(label="Generated Images")
            browse_lora_btn.click(fn=load_lora, inputs=[], outputs=lora_path)
            generate_btn.click(
                fn=generate_image,
                inputs=[scheduler_name, lora_path, lora_scale, width, height, 
                        prompt, negative_prompt, steps, guidance, 
                        seed, batch_count, batch_size],
                outputs=output_gallery
            )

        
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