# 🔧 LoRA Trainer GUI for SD 1.5 — Based on `bmaltis/kohya_ss`

This GUI is a lightweight, user-friendly interface for training **LoRA models** for Stable Diffusion 1.5. It builds upon the power of [`bmaltis/kohya_ss`](https://github.com/bmaltais/kohya_ss) and wraps it with an intuitive frontend using [Gradio](https://gradio.app).

> ⚠️ This tool is intended for **Windows** users and requires installation **directly in your C drive** (e.g., `C:\lora-trainer`).

---

## ⚙️ Windows Pre-requirements

Please make sure the following are installed before setup:

1. ✅ **[Python 3.10.11](https://www.python.org/downloads/release/python-31011/)**
    - ⚠️ Make sure to **check the box** to add Python to the **`PATH`** during installation.

2. ✅ **[CUDA Toolkit 12.4](https://developer.nvidia.com/cuda-downloads)**

3. ✅ **[Git for Windows](https://git-scm.com/)**

4. ✅ **[Microsoft Visual C++ Redistributable (2015–2022)](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)**

---

## 📦 Installation

1. Open Command Prompt and run:

   ```bash
   git clone https://github.com/xin-04/lora-trainer

2. Run the setup scripts in order:
   i. run step1_setup_venv.bat
   ii. run step2_download_models.bat
   iii. run afterinstall_run_gui.bat

## 🥊Main features

This GUI provides three main workflows for SD 1.5 fine-tuning:

1. Dataset Preparation
   - Upload and crop training images
   - Resize images to standard SD resolutions (512x512, 768x512, etc.)
   - Generate automatic captions for dataset images
  
2. LoRA Training
   - Customize LoRA training settings
   - Load and preview configurations
   - Train models using your own dataset and Hugging Face models
  
3. Model Deployment
   - Generate images using trained LoRA models
   - Adjust inference parameters
   - Visual preview of generated results
 
