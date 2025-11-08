d launch_chatbot.bat
@echo off
title Local AI Chatbot Launcher
color 0A

echo ========================================
echo    Local AI Chatbot (v2.0)
echo ========================================
echo.

cd /d "C:\Users\matei\llama-chatbot"

echo [INFO] Checking Python installation...
python --version
if errorlevel 1 (
    echo [ERROR] Python not found!
    pause
    exit /b 1
)

echo.
echo [INFO] Starting chatbot...
echo [INFO] If there are errors, they will appear below:
echo.

python app/main.py

echo.
echo ========================================
echo Chatbot closed.
echo ========================================
echo.
echo If the window closed immediately, there was a Python error above.
echo.
pause