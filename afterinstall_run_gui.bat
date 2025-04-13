@echo off

cd /d "C:\lora-trainer"

call .\venv\Scripts\activate.bat

cd "GUI"

python main.py
