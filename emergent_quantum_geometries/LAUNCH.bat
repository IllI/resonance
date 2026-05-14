@echo off
REM JILA Pipeline GUI Launcher (Windows)
REM Double-click this file, or run it from any terminal.
REM
REM Requirements: Python 3.9+ must be installed.
REM   Download from: https://www.python.org/downloads/
REM   (Check "Add Python to PATH" during install)
REM
REM Everything else (numpy, scipy, matplotlib, etc.) is installed
REM automatically into a local .venv\ folder — nothing goes global.

python launch.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Launch failed. Make sure Python 3.9+ is installed.
    echo Download from: https://www.python.org/downloads/
    pause
)
