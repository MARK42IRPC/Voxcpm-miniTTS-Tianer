@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

if /I "%~1"=="--check" (
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" -Profile recommended -Check
    exit /b %errorlevel%
)

echo VoxCPM miniTTS first-time installer
echo Dependencies and selected models are skipped when already complete.
echo.

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
if errorlevel 1 (
    echo.
    echo Installation failed. Review the message above, then run this file again.
    pause
    exit /b 1
)

echo.
echo Starting VoxCPM miniTTS WebUI...
call "%~dp0start_webui.bat"
