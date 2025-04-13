@echo off
echo Downloading model from Hugging Face...

set "TARGET_DIR=C:\lora-trainer\models\unet"

if not exist "%TARGET_DIR%" (
    mkdir "%TARGET_DIR%"
)

curl -L -o "%TARGET_DIR%\v1-5-pruned.safetensors" "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned.safetensors?download=true"

echo Download complete!
echo File saved to: %TARGET_DIR%

set "VAE_TARGET_DIR=C:\lora-trainer\models\vae"

if not exist "%VAE_TARGET_DIR%" (
    mkdir "%VAE_TARGET_DIR%"
)

curl -L -o "%VAE_TARGET_DIR%\diffusion_pytorch_model.safetensors" "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.safetensors?download=true"

pause
