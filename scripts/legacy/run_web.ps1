param(
    [string]$HostAddress = "127.0.0.1",
    [ValidateRange(1, 65535)]
    [int]$Port = 8810
)

$ErrorActionPreference = "Stop"
$Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$CacheRoot = "C:\tmp\voxcpm"
$env:TRITON_HOME = Join-Path $CacheRoot "triton-home"
$env:TRITON_CACHE_DIR = Join-Path $CacheRoot "triton-cache"
$env:TORCHINDUCTOR_CACHE_DIR = Join-Path $CacheRoot "inductor-cache"
$env:HF_HOME = Join-Path $CacheRoot "hf-cache"
$env:TEMP = Join-Path $CacheRoot "temp"
$env:TMP = $env:TEMP
$env:TRITON_HOME, $env:TRITON_CACHE_DIR, $env:TORCHINDUCTOR_CACHE_DIR, $env:HF_HOME, $env:TEMP |
    ForEach-Object { New-Item -ItemType Directory -Force -Path $_ | Out-Null }

if (-not (Test-Path -LiteralPath $Python)) {
    throw "VoxCPM is not installed. Expected Python: $Python"
}

& $Python -X utf8 (Join-Path $PSScriptRoot "webui.py") --host $HostAddress --port $Port

if ($LASTEXITCODE -ne 0) {
    throw "VoxCPM web server failed with exit code $LASTEXITCODE"
}
