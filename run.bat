@echo off
title Autocomplete Engine
echo ============================================
echo   Multilingual Autocomplete Engine Launcher
echo ============================================
echo.

REM Check if .venv exists, create if not
if not exist ".venv" (
    echo [1/3] Creating virtual environment...
    python -m venv .venv
) else (
    echo [1/3] Virtual environment found.
)

REM Install dependencies
echo [2/3] Installing dependencies...
.venv\Scripts\pip install -r requirements.txt -q

REM Start the server
echo [3/3] Starting server...
echo.
echo ============================================
echo   Server running at http://127.0.0.1:8000
echo   Press Ctrl+C to stop
echo ============================================
echo.
.venv\Scripts\python -m uvicorn autocomplete_engine.api.server:app --reload --port 8000

pause
