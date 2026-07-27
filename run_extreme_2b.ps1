param(
    [string]$Text = "你好，欢迎使用 VoxCPM。",
    [string]$ReferenceAudio = "pretrained_models\ZipEnhancer\examples\speech_with_noise.wav",
    [string]$Output = "outputs\extreme_2b.wav",
    [ValidateRange(1, 100)]
    [int]$InferenceTimesteps = 4,
    [int]$Seed = 42
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$VoxCpm = Join-Path $ProjectRoot ".venv\Scripts\voxcpm.exe"
$ModelPath = Join-Path $ProjectRoot "pretrained_models\VoxCPM2"
$DenoiserPath = Join-Path $ProjectRoot "pretrained_models\ZipEnhancer"
$ReferencePath = if ([System.IO.Path]::IsPathRooted($ReferenceAudio)) {
    $ReferenceAudio
} else {
    Join-Path $ProjectRoot $ReferenceAudio
}
$OutputPath = if ([System.IO.Path]::IsPathRooted($Output)) {
    $Output
} else {
    Join-Path $ProjectRoot $Output
}

$CacheRoot = "C:\tmp\voxcpm"
$env:TRITON_HOME = Join-Path $CacheRoot "triton-home"
$env:TRITON_CACHE_DIR = Join-Path $CacheRoot "triton-cache"
$env:TORCHINDUCTOR_CACHE_DIR = Join-Path $CacheRoot "inductor-cache"
$env:TEMP = Join-Path $CacheRoot "temp"
$env:TMP = $env:TEMP
$env:TRITON_HOME, $env:TRITON_CACHE_DIR, $env:TORCHINDUCTOR_CACHE_DIR, $env:TEMP |
    ForEach-Object { New-Item -ItemType Directory -Force -Path $_ | Out-Null }

if (-not (Test-Path -LiteralPath $VoxCpm)) {
    throw "VoxCPM is not installed: $VoxCpm"
}
if (-not (Test-Path -LiteralPath (Join-Path $ModelPath "model.safetensors"))) {
    throw "VoxCPM2 is not installed: $ModelPath"
}
if (-not (Test-Path -LiteralPath $ReferencePath)) {
    throw "Reference audio is missing: $ReferencePath"
}

& $VoxCpm clone `
    --text $Text `
    --reference-audio $ReferencePath `
    --output $OutputPath `
    --model-path $ModelPath `
    --device cuda `
    --zipenhancer-path $DenoiserPath `
    --denoise `
    --inference-timesteps $InferenceTimesteps `
    --seed $Seed

if ($LASTEXITCODE -ne 0) {
    throw "VoxCPM2 extreme inference failed with exit code $LASTEXITCODE"
}
