@echo off
echo 🎓 Face Recognition Attendance System - GUI Launcher
echo =====================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo    Please install Python 3.7+ and try again
    pause
    exit /b 1
)

echo ✅ Python found
echo.

REM Launch the GUI launcher
echo 🚀 Starting GUI application...
python launch_gui.py

REM Keep window open if there was an error
if errorlevel 1 (
    echo.
    echo ❌ Application exited with an error
    pause
)
