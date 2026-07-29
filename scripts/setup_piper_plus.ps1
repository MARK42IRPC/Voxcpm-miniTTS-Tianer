[CmdletBinding()]
param(
    [string]$UvExe = "",
    [switch]$Check
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$PiperPlusVersion = "1.13.0"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$SourceRoot = Join-Path $RepoRoot "third_party\piper-plus"
$VenvRoot = Join-Path $RepoRoot ".venv-piper-plus"
$MainPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$PlusPython = Join-Path $VenvRoot "Scripts\python.exe"

function Resolve-Uv {
    if ($UvExe -and (Test-Path -LiteralPath $UvExe)) {
        return (Resolve-Path -LiteralPath $UvExe).Path
    }
    $command = Get-Command uv -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }
    $bundled = Join-Path $RepoRoot ".tools\uv\uv.exe"
    if (Test-Path -LiteralPath $bundled) {
        return $bundled
    }
    throw "uv is not installed. Run install.ps1 first."
}

function Test-PiperPlusEnvironment {
    if (-not (Test-Path -LiteralPath $PlusPython)) {
        return $false
    }
    $required = @(
        (Join-Path $SourceRoot "src\python\piper_train\__main__.py"),
        (Join-Path $SourceRoot "src\python\g2p\piper_plus_g2p\__init__.py"),
        (Join-Path $SourceRoot "src\python_run\piper\voice.py")
    )
    foreach ($path in $required) {
        if (-not (Test-Path -LiteralPath $path)) {
            return $false
        }
    }
    & $PlusPython -c "import torch, onnxruntime; from piper import PiperVoice; from piper_train.__main__ import create_parser; from piper_plus_g2p.multilingual import MultilingualPhonemizer" 2>$null
    return $LASTEXITCODE -eq 0
}

if ($Check) {
    $ready = Test-PiperPlusEnvironment
    Write-Host "Piper Plus environment: $(if ($ready) { 'ready' } else { 'not installed' })"
    Write-Host "Piper Plus source: $SourceRoot"
    Write-Host "Piper Plus Python: $PlusPython"
    if (-not $ready) {
        exit 1
    }
    exit 0
}

if (-not (Test-Path -LiteralPath $MainPython)) {
    throw "Main Python environment is missing: $MainPython"
}

if (-not (Test-Path -LiteralPath $SourceRoot)) {
    $git = Get-Command git -ErrorAction SilentlyContinue
    if (-not $git) {
        throw "Git is required to install the pinned Piper Plus training source."
    }
    Write-Host "==> Downloading Piper Plus v$PiperPlusVersion training source" -ForegroundColor Cyan
    $sourceParent = Split-Path -Parent $SourceRoot
    New-Item -ItemType Directory -Force -Path $sourceParent | Out-Null
    & $git.Source clone --depth 1 --branch "v$PiperPlusVersion" "https://github.com/ayutaz/piper-plus.git" $SourceRoot
    if ($LASTEXITCODE -ne 0) {
        & $git.Source clone --depth 1 --branch "v$PiperPlusVersion" "https://ghfast.top/https://github.com/ayutaz/piper-plus.git" $SourceRoot
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Piper Plus source download failed."
    }
}

$versionFile = Join-Path $SourceRoot "VERSION"
if (-not (Test-Path -LiteralPath $versionFile)) {
    throw "Piper Plus source is incomplete: VERSION is missing."
}
$installedVersion = (Get-Content -LiteralPath $versionFile -Raw).Trim()
if ($installedVersion -ne $PiperPlusVersion) {
    throw "Piper Plus source version is $installedVersion; expected $PiperPlusVersion. Move the existing source directory aside and rerun setup."
}

$ResolvedUv = Resolve-Uv
if (-not (Test-Path -LiteralPath $PlusPython)) {
    Write-Host "==> Creating isolated Piper Plus environment" -ForegroundColor Cyan
    & $ResolvedUv venv --python $MainPython --system-site-packages $VenvRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Piper Plus environment creation failed with exit code $LASTEXITCODE."
    }
}

# A venv with system-site-packages only sees the base interpreter's global
# packages, not packages from the sibling main .venv. Prepend the three Piper
# Plus source roots so its `piper` package wins over the main piper-tts package,
# then expose the main environment for the shared Torch/CUDA stack.
$PlusSitePackages = Join-Path $VenvRoot "Lib\site-packages"
$MainSitePackages = Join-Path $RepoRoot ".venv\Lib\site-packages"
$RuntimeSource = Join-Path $SourceRoot "src\python_run"
$TrainSource = Join-Path $SourceRoot "src\python"
$G2pSource = Join-Path $SourceRoot "src\python\g2p"
$BridgePath = Join-Path $PlusSitePackages "voxcpm_piper_plus_paths.pth"
$BridgeCode = "import sys; sys.path[:0] = [r`"$RuntimeSource`", r`"$TrainSource`", r`"$G2pSource`", r`"$MainSitePackages`"]"
[IO.File]::WriteAllText($BridgePath, $BridgeCode + [Environment]::NewLine, [Text.UTF8Encoding]::new($false))

Write-Host "==> Installing Piper Plus runtime and training adapters" -ForegroundColor Cyan
$G2pProject = Join-Path $SourceRoot "src\python\g2p"
$TrainProject = Join-Path $SourceRoot "src\python"
$RuntimeProject = Join-Path $SourceRoot "src\python_run"
& $ResolvedUv pip install --python $PlusPython --no-deps --editable $G2pProject --editable $TrainProject --editable $RuntimeProject
if ($LASTEXITCODE -ne 0) {
    throw "Piper Plus editable package installation failed with exit code $LASTEXITCODE."
}

$ExtraDependencies = @(
    "numpy>=1.26.4,<2.5",
    "pyopenjtalk-plus>=0.4.1.post8,<1",
    "mecab-python3>=1.0",
    "unidic-lite>=1.0",
    "seaborn>=0.13",
    "onnxsim-prebuilt",
    "onnxscript>=0.6.2"
)
& $ResolvedUv pip install --python $PlusPython @ExtraDependencies
if ($LASTEXITCODE -ne 0) {
    throw "Piper Plus support dependency installation failed with exit code $LASTEXITCODE."
}

if (-not (Test-PiperPlusEnvironment)) {
    throw "Piper Plus installation completed, but its runtime modules cannot be imported."
}

Write-Host "Piper Plus v$PiperPlusVersion environment is ready." -ForegroundColor Green
