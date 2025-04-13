@echo off
cd /d "C:\lora-trainer"
call .\venv\Scripts\activate.bat
cd "sd-scripts"
accelerate launch train_network.py --config_file "%1"