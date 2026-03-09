# Download GigaAM v3 CTC ONNX model (sherpa-onnx format) for Russian ASR
# Source: https://huggingface.co/csukuangfj/sherpa-onnx-nemo-ctc-giga-am-v3-russian-2025-12-16

$ErrorActionPreference = "Stop"
$BaseUrl = "https://huggingface.co/csukuangfj/sherpa-onnx-nemo-ctc-giga-am-v3-russian-2025-12-16/resolve/main"
$OutDir = Join-Path (Join-Path $PSScriptRoot "..") (Join-Path "models" "gigaam-v3-ctc-ru")

if (-not (Test-Path $OutDir)) {
    New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
}

$Files = @(
    @{ Name = "model.int8.onnx"; Url = "$BaseUrl/model.int8.onnx" },
    @{ Name = "tokens.txt"; Url = "$BaseUrl/tokens.txt" }
)

foreach ($f in $Files) {
    $path = Join-Path $OutDir $f.Name
    if (Test-Path $path) {
        Write-Host "Already exists: $path"
    } else {
        Write-Host "Downloading $($f.Name)..."
        Invoke-WebRequest -Uri $f.Url -OutFile $path -UseBasicParsing
        Write-Host "Saved: $path"
    }
}

Write-Host "Done. Model dir: $OutDir"
