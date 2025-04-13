@echo off

IF NOT EXIST venv (
    echo Creating venv...
    python -m venv venv
)

:: Create the directory if it doesn't exist
mkdir ".\logs\setup" > nul 2>&1

:: Deactivate the virtual environment to prevent error
call .\venv\Scripts\deactivate.bat

call .\venv\Scripts\activate.bat

REM first make sure we have setuptools available in the venv    
python -m pip install --require-virtualenv --no-input -q -q  setuptools

:: Install required modules
pip install -r requirements.txt

:: Deactivate the virtual environment
call .\venv\Scripts\deactivate.bat