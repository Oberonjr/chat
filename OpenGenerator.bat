@echo off
echo ================================
echo Local AI Chatbot Launcher
echo ================================
echo.

cd /d "C:\Users\matei\llama-chatbot"

echo Starting chatbot...
echo.

REM Run as a Python module
python -m app.main

echo.
echo Chatbot closed.
pause