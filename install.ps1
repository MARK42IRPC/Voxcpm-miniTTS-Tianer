[CmdletBinding()]
param(
    [ValidateSet("prompt", "none", "lite", "recommended", "full")]
    [string]$Profile = "prompt",
    [string]$HfEndpoint = "",
    [switch]$UseChinaMirror,
    [switch]$ForceModels,
    [switch]$Check
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$RepoRoot = $PSScriptRoot
$ToolsRoot = Join-Path $RepoRoot ".tools"
$UvRoot = Join-Path $ToolsRoot "uv"
$UvExe = Join-Path $UvRoot "uv.exe"
$CacheRoot = if ($env:VOXCPM_CACHE_DIR) { $env:VOXCPM_CACHE_DIR } else { "C:\tmp\voxcpm" }

function Write-Step([string]$Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Resolve-Uv {
    $command = Get-Command uv -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }
    if (Test-Path -LiteralPath $UvExe) {
        return $UvExe
    }
    if ($Check) {
        return $null
    }

    Write-Step "Downloading the uv package manager"
    $archive = Join-Path $ToolsRoot "uv-windows.zip"
    New-Item -ItemType Directory -Force -Path $ToolsRoot, $UvRoot | Out-Null
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $uvUrl = "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip"
    try {
        Invoke-WebRequest -UseBasicParsing -Uri $uvUrl -OutFile $archive
    }
    catch {
        Write-Warning "Direct GitHub download failed; retrying through ghfast."
        Invoke-WebRequest -UseBasicParsing -Uri "https://ghfast.top/$uvUrl" -OutFile $archive
    }
    Expand-Archive -LiteralPath $archive -DestinationPath $UvRoot -Force
    Remove-Item -LiteralPath $archive -Force
    if (-not (Test-Path -LiteralPath $UvExe)) {
        $candidate = Get-ChildItem -LiteralPath $UvRoot -Recurse -Filter "uv.exe" -File | Select-Object -First 1
        if (-not $candidate) {
            throw "The uv archive did not contain uv.exe"
        }
        $script:UvExe = $candidate.FullName
    }
    return $script:UvExe
}

function Select-Profile {
    if ($Profile -ne "prompt") {
        return $Profile
    }
    Write-Host ""
    Write-Host "Select an installation profile:" -ForegroundColor White
    Write-Host "  1. Lite        - VoxCPM 0.5B + denoiser + one Piper voice"
    Write-Host "  2. Recommended - VoxCPM2 + 0.5B + denoiser + two Piper voices"
    Write-Host "  3. Full        - all VoxCPM models + denoiser + Piper voices + Melo training base"
    Write-Host "  4. Dependencies only"
    $choice = Read-Host "Choice [2]"
    switch ($choice.Trim()) {
        "1" { return "lite" }
        "3" { return "full" }
        "4" { return "none" }
        default { return "recommended" }
    }
}

Set-Location -LiteralPath $RepoRoot
$SelectedProfile = Select-Profile
if ($UseChinaMirror -and -not $HfEndpoint) {
    $HfEndpoint = "https://hf-mirror.com"
}

$env:VOXCPM_CACHE_DIR = $CacheRoot
$env:TRITON_HOME = Join-Path $CacheRoot "triton-home"
$env:TRITON_CACHE_DIR = Join-Path $CacheRoot "triton-cache"
$env:TORCHINDUCTOR_CACHE_DIR = Join-Path $CacheRoot "inductor-cache"
$env:HF_HOME = Join-Path $CacheRoot "hf-cache"
$env:HF_HUB_DISABLE_XET = "1"
$env:TEMP = Join-Path $CacheRoot "temp"
$env:TMP = $env:TEMP
if ($HfEndpoint) {
    $env:HF_ENDPOINT = $HfEndpoint.TrimEnd("/")
}
$env:TRITON_HOME, $env:TRITON_CACHE_DIR, $env:TORCHINDUCTOR_CACHE_DIR, $env:HF_HOME, $env:TEMP |
    ForEach-Object { New-Item -ItemType Directory -Force -Path $_ | Out-Null }

$ResolvedUv = Resolve-Uv
if ($Check) {
    Write-Host "Repository: $RepoRoot"
    Write-Host "Profile: $SelectedProfile"
    Write-Host "Cache: $CacheRoot"
    Write-Host "uv: $(if ($ResolvedUv) { $ResolvedUv } else { 'not installed (will be downloaded)' })"
    Write-Host "Installer check passed."
    exit 0
}

$EstimatedModelGB = switch ($SelectedProfile) {
    "lite" { 2 }
    "recommended" { 8 }
    "full" { 10 }
    default { 0 }
}
$DriveName = [IO.Path]::GetPathRoot($RepoRoot).TrimEnd("\").TrimEnd(":")
$Drive = Get-PSDrive -Name $DriveName -ErrorAction SilentlyContinue
if ($Drive) {
    $FreeGB = [math]::Round($Drive.Free / 1GB, 1)
    Write-Host "Disk space: $FreeGB GB free; model profile needs about $EstimatedModelGB GB plus dependencies/cache."
}

Write-Step "Creating the Python 3.12 environment and installing locked dependencies"
& $ResolvedUv sync --frozen --python 3.12 --extra dev
if ($LASTEXITCODE -ne 0) {
    throw "Dependency installation failed with exit code $LASTEXITCODE"
}

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Dependency installation completed without creating $Python"
}

Write-Step "Installing model profile: $SelectedProfile"
$ModelArgs = @("-X", "utf8", "scripts/install_models.py", "--profile", $SelectedProfile)
if ($ForceModels) {
    $ModelArgs += "--force"
}
if ($HfEndpoint) {
    $ModelArgs += @("--hf-endpoint", $HfEndpoint)
}
& $Python @ModelArgs
if ($LASTEXITCODE -ne 0) {
    throw "Model installation failed with exit code $LASTEXITCODE"
}

Write-Step "Installation complete"
Write-Host "Use start_webui.bat for future launches."
Write-Host "WebUI address: http://127.0.0.1:8810"
