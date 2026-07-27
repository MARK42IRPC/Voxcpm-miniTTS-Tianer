@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo VoxCPM virtual environment was not found.
    echo Run install_and_start.bat for the first installation.
    pause
    exit /b 1
)

set "VOXCPM_CACHE_DIR=C:\tmp\voxcpm"
set "TRITON_HOME=%VOXCPM_CACHE_DIR%\triton-home"
set "TRITON_CACHE_DIR=%VOXCPM_CACHE_DIR%\triton-cache"
set "TORCHINDUCTOR_CACHE_DIR=%VOXCPM_CACHE_DIR%\inductor-cache"
set "HF_HOME=%VOXCPM_CACHE_DIR%\hf-cache"
set "TEMP=%VOXCPM_CACHE_DIR%\temp"
set "TMP=%TEMP%"
set "VOXCPM_CPU_THREADS=8"
set "CUBLAS_WORKSPACE_CONFIG=:4096:8"
set "VOXCPM_HYBRID_DETERMINISTIC=1"

if not exist "%TRITON_HOME%" mkdir "%TRITON_HOME%"
if not exist "%TRITON_CACHE_DIR%" mkdir "%TRITON_CACHE_DIR%"
if not exist "%TORCHINDUCTOR_CACHE_DIR%" mkdir "%TORCHINDUCTOR_CACHE_DIR%"
if not exist "%TEMP%" mkdir "%TEMP%"

if /I "%~1"=="--check" (
    echo VoxCPM WebUI launcher is ready.
    exit /b 0
)

powershell.exe -NoProfile -Command "if (Get-NetTCPConnection -State Listen -LocalPort 8810 -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"
if not errorlevel 1 (
    start "" "http://127.0.0.1:8810"
    exit /b 0
)

start "" powershell.exe -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 4; Start-Process 'http://127.0.0.1:8810'"

echo Starting VoxCPM WebUI at http://127.0.0.1:8810
echo Close this window to stop the server.
echo.
".venv\Scripts\python.exe" -X utf8 webui.py --host 127.0.0.1 --port 8810

if errorlevel 1 (
    echo.
    echo VoxCPM WebUI stopped with an error.
    pause
)
