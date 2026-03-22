# Generate self-signed TLS certs for local WSS development.
# Usage: .\tools\gen-dev-certs.ps1 [-OutDir .\certs]
# Default output: .\certs\
# Requires: OpenSSL (via Git Bash, WSL, or standalone installation)

param(
    [string]$OutDir = ".\certs"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $OutDir)) {
    New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
}

# Try to find OpenSSL
$openssl = $null
$opensslPaths = @(
    "openssl",
    "$env:ProgramFiles\Git\usr\bin\openssl.exe",
    "$env:ProgramFiles(x86)\Git\usr\bin\openssl.exe",
    "C:\Program Files\OpenSSL-Win64\bin\openssl.exe",
    "C:\OpenSSL-Win64\bin\openssl.exe"
)

foreach ($path in $opensslPaths) {
    if (Get-Command $path -ErrorAction SilentlyContinue) {
        $openssl = $path
        break
    }
}

if (-not $openssl) {
    Write-Error "OpenSSL not found. Please install OpenSSL or use Git Bash/WSL."
    Write-Host "Options:"
    Write-Host "  1. Install Git for Windows (includes OpenSSL)"
    Write-Host "  2. Install OpenSSL from https://slproweb.com/products/Win32OpenSSL.html"
    Write-Host "  3. Use WSL: wsl bash ./tools/gen-dev-certs.sh"
    exit 1
}

$certPath = Join-Path $OutDir "server.pem"
$keyPath = Join-Path $OutDir "server.key"

# Minimal openssl.cnf: MSYS2/Git OpenSSL often has OPENSSL_CONF pointing at a missing file
# (e.g. E:/msys64/mingw64/etc/ssl/openssl.cnf). We ignore that and use our own config.
$cnfPath = Join-Path $env:TEMP "localvox-openssl-$PID.cnf"
$cnfContent = @"
[req]
distinguished_name = req_distinguished_name
x509_extensions = v3_req
prompt = no

[req_distinguished_name]
CN = localhost

[v3_req]
subjectAltName = @alt_names

[alt_names]
DNS.1 = localhost
DNS.2 = 127.0.0.1
IP.1 = 127.0.0.1
"@
Set-Content -LiteralPath $cnfPath -Value $cnfContent -Encoding ascii

$oldOpensslConf = $env:OPENSSL_CONF
try {
    Remove-Item Env:OPENSSL_CONF -ErrorAction SilentlyContinue
    & $openssl req -x509 -newkey rsa:4096 -keyout $keyPath -out $certPath `
        -days 365 -nodes `
        -subj "/CN=localhost" `
        -config $cnfPath
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to generate certificates"
        exit 1
    }
}
finally {
    if ($null -ne $oldOpensslConf -and $oldOpensslConf -ne "") {
        $env:OPENSSL_CONF = $oldOpensslConf
    }
    Remove-Item -LiteralPath $cnfPath -Force -ErrorAction SilentlyContinue
}

Write-Host "Created: $certPath, $keyPath"
Write-Host ""
Write-Host "Add to settings.json:"
Write-Host "  `"tls_cert_path`": `"$certPath`","
Write-Host "  `"tls_key_path`": `"$keyPath`""
Write-Host ""
Write-Host "Client: use wss://localhost:9745 and --tls-insecure (or add server.pem as custom CA)"
