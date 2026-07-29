param(
    [Parameter(Mandatory = $true)]
    [string]$Text,

    [string]$Output = "outputs\voxcpm.wav",

    [ValidateRange(1, 100)]
    [int]$InferenceTimesteps = 10,

    [int]$Seed = 42
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$VoxCpm = Join-Path $ProjectRoot ".venv\Scripts\voxcpm.exe"
$ModelPath = Join-Path $ProjectRoot "pretrained_models\VoxCPM-0.5B"
$CacheRoot = "C:\tmp\voxcpm"
$env:TRITON_HOME = Join-Path $CacheRoot "triton-home"
$env:TRITON_CACHE_DIR = Join-Path $CacheRoot "triton-cache"
$env:TORCHINDUCTOR_CACHE_DIR = Join-Path $CacheRoot "inductor-cache"
$env:TEMP = Join-Path $CacheRoot "temp"
$env:TMP = $env:TEMP
$env:TRITON_HOME, $env:TRITON_CACHE_DIR, $env:TORCHINDUCTOR_CACHE_DIR, $env:TEMP |
    ForEach-Object { New-Item -ItemType Directory -Force -Path $_ | Out-Null }
$OutputPath = if ([System.IO.Path]::IsPathRooted($Output)) {
    $Output
} else {
    Join-Path $ProjectRoot $Output
}

if (-not (Test-Path -LiteralPath $VoxCpm)) {
    throw "VoxCPM is not installed. Expected CLI: $VoxCpm"
}

if (-not (Test-Path -LiteralPath (Join-Path $ModelPath "pytorch_model.bin"))) {
    throw "VoxCPM-0.5B weights are missing. Expected model: $ModelPath"
}

& $VoxCpm design `
    --text $Text `
    --output $OutputPath `
    --model-path $ModelPath `
    --device cuda `
    --no-denoiser `
    --no-optimize `
    --inference-timesteps $InferenceTimesteps `
    --seed $Seed

if ($LASTEXITCODE -ne 0) {
    throw "VoxCPM inference failed with exit code $LASTEXITCODE"
}
