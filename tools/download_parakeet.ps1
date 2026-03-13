# Download Parakeet TDT 0.6B v3 (sherpa-onnx format) for Russian + 24 European languages
# Source: https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2

$ErrorActionPreference = "Stop"
$ArchiveUrl = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2"
$RootDir = Split-Path $PSScriptRoot -Parent
$OutDir = Join-Path $RootDir (Join-Path "models" "parakeet-tdt-0.6b-v3-int8")
$ArchivePath = Join-Path $env:TEMP "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2"

if (-not (Test-Path (Join-Path $RootDir "models"))) {
    New-Item -ItemType Directory -Path (Join-Path $RootDir "models") -Force | Out-Null
}

# Check if already extracted
$encoderPath = Join-Path $OutDir "encoder.int8.onnx"
if (Test-Path $encoderPath) {
    Write-Host "Parakeet model already exists: $OutDir"
    exit 0
}

Write-Host "Downloading Parakeet TDT 0.6B v3 (~640 MB)..."
Invoke-WebRequest -Uri $ArchiveUrl -OutFile $ArchivePath -UseBasicParsing

Write-Host "Extracting..."
New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
tar -xjf $ArchivePath -C $OutDir --strip-components=1

Remove-Item $ArchivePath -Force -ErrorAction SilentlyContinue
Write-Host "Done. Model dir: $OutDir"
Write-Host "Add to settings.json models: { `"type`": `"parakeet`", `"model_path`": `"models/parakeet-tdt-0.6b-v3-int8`" }"
