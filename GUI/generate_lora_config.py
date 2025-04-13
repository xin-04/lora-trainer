import toml
import os
from tkinter import Tk, filedialog

def save_to_toml(data):
    default_dir = os.path.abspath("C:\lora-trainer\lora_presets")
    root = Tk()
    root.withdraw()

    # Ask user where to save the file
    file_path = filedialog.asksaveasfilename(
        initialdir = default_dir,
        defaultextension=".toml",
        filetypes=[("TOML files", "*.toml")],
        title="Save configuration as..."
    )

    if file_path:
        with open(file_path, "w") as toml_file:
            toml.dump(data, toml_file)
        print(f"Configuration saved to: {file_path}")
    else:
        print("Save cancelled.")

def save_dataset_to_toml(data):
    file_path = os.path.abspath("C:\lora-trainer\sd-scripts")
    root = Tk()
    root.withdraw()

    if file_path:
        with open(file_path, "w") as toml_file:
            toml.dump(data, toml_file)
        print(f"Configuration saved to: {file_path}")
    else:
        print("Save cancelled.")

def save_config(
    model_path, vae_path, train_data_dir, output_dir,
    resolution, bucket_reso_steps, network_dim, network_alpha, seed,
    learning_rate, train_batch_size, max_train_steps, max_train_epochs,
    output_name, save_model_as, save_precision, mixed_precision,
    huber_schedule, huber_c, adaptive_noise_scale, prior_loss_weight, loss_type,
    vae_batch_size, max_data_loader_n_workers, persistent_data_loader_workers,
    unet_lr, text_encoder_lr, lr_scheduler, lr_scheduler_num_cycles, lr_scheduler_power,
    max_token_length, max_timestep, sample_sampler, sample_every_n_epochs,
    save_every_n_steps, save_every_n_epochs, gradient_accumulation_steps, clip_skip,
    multires_noise_discount, min_snr_gamma, noise_offset, highvram, lowram, xformers, sdpa, 
    bucket_no_upscale, cache_latents, cache_latents_to_disk,
    optimizer_type, optimizer_args, use_8bit_adam, use_lion_optimizer,
    network_train_unet_only, network_train_text_encoder_only, dim_from_weights,
    no_half_vae, fp8_base, full_bf16, full_fp16,
    gradient_checkpointing, skip_cache_check, enable_bucket
):
    config = {
        "model": {
            "pretained_model_name_or_path": model_path,
            "output_name": output_name,
            "train_data_dir": train_data_dir,
            "dataset_config": "dataset.toml",
            "save_model_as": save_model_as
            
        },
        "folders": {
            "output_dir": output_dir,
            "logging_dir": "./logs",
            "reg_data_dir": "./data/reg",
        },
        "configuration":{
            "config_dir": "./lora_presets",
        },
        "accelerate_launch":{
            "mixed_precision": mixed_precision,               
        },
        "basic":{
            "cache_latents": cache_latents,                  
            "cache_latents_to_disk": cache_latents_to_disk,        
            "caption_extension": ".txt",           
            "enable_bucket": enable_bucket,                 
            "learning_rate": learning_rate,                
            "lr_scheduler": lr_scheduler,               
            "lr_scheduler_num_cycles": lr_scheduler_num_cycles,           
            "lr_scheduler_power": lr_scheduler_power,
            "network_dim": network_dim,
            "network_alpha": network_alpha,
            "max_bucket_reso": 2048,                
            "max_grad_norm": 1.0,                   
            "max_resolution": resolution,            
            "max_train_steps": max_train_steps,                   
            "max_train_epochs": max_train_epochs,                  
            "min_bucket_reso": bucket_reso_steps,                 
            "optimizer_type": optimizer_type,               
            "optimizer_args": optimizer_args,       
            "use_8bit_adam": use_8bit_adam,
            "use_lion_optimizer": use_lion_optimizer,            
            "save_every_n_epochs": save_every_n_epochs,               
            "save_every_n_steps": save_every_n_steps, 
            "save_precision": save_precision,               
            "seed": seed,                          
            "stop_text_encoder_training": 0,        
            "train_batch_size": train_batch_size,     
            "text_encoder_lr": text_encoder_lr,
            "unet_lr": unet_lr,                
        },
        "advanced":{
            "adaptive_noise_scale": adaptive_noise_scale,                          
            "additional_parameters": "",                        
            "bucket_no_upscale": bucket_no_upscale,                          
            "bucket_reso_steps": bucket_reso_steps,                            
            "clip_skip": clip_skip,                                     
            "dim_from_weights": dim_from_weights,            
            "fp8_base": fp8_base,                                  
            "full_bf16": full_bf16,                                 
            "full_fp16": full_fp16,                                 
            "gradient_accumulation_steps": gradient_accumulation_steps,                   
            "gradient_checkpointing": gradient_checkpointing,     
            "highvram": highvram,               
            "huber_c": huber_c,                                     
            "huber_schedule": huber_schedule,                            
            "log_tracker_config_dir": "./logs",                 
            "loss_type": loss_type,    
            "lowram": lowram,                              
            "max_data_loader_n_workers": max_data_loader_n_workers,                     
            "max_timestep": max_timestep,                               
            "max_token_length": max_token_length,                            
            "min_snr_gamma": min_snr_gamma,                                 
            "multires_noise_discount": multires_noise_discount,  
            "network_train_unet_only": network_train_unet_only,
            "network_train_text_encoder_only": network_train_text_encoder_only,
            "no_half_vae": no_half_vae,                     
            "noise_offset": noise_offset,                                  
            "persistent_data_loader_workers": persistent_data_loader_workers,            
            "prior_loss_weight": prior_loss_weight,                           
            "save_every_n_steps": save_every_n_steps,                            
            "skip_cache_check": skip_cache_check,                   
            "state_dir": "./outputs",                           
            "log_with": "tensorboard",                                     
            "vae_batch_size": vae_batch_size,                                
            "vae": vae_path,                          
            "xformers": xformers,  
            "sdpa": sdpa                            
        },
        "dataset_preparation": {
            "images_folder": train_data_dir,
        },
        "samples":{
            "sample_every_n_steps": sample_every_n_epochs,
            "sample_sampler": sample_sampler
        }
    }

    save_to_toml(config)

    dataset_config={
        "general":{
            "caption_extension": ".txt",
            "keep_tokens": 1,
        },
        "datasets": [
        {
            "resolution": resolution,
            "batch_size": train_batch_size,
            "enable_bucket": enable_bucket,
            "subsets": [
                {
                    "image_dir": train_data_dir, 
                    "class_tokens": "",
                    "num_repeats": 10,
                }
            ],
        }
    ]
    }

    save_dataset_to_toml(dataset_config)

    try:
        with open("config_saved.toml", "w") as f:
            toml.dump(config, f)
        return "✅ Config saved to config_saved.toml"
    except Exception as e:
        return f"❌ Failed to save config: {str(e)}"
    

    