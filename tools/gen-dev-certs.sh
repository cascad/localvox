#!/bin/bash
# Generate self-signed TLS certs for local WSS development.
# Usage: ./tools/gen-dev-certs.sh [output_dir]
# Default output: ./certs/

set -e
OUT_DIR="${1:-./certs}"
mkdir -p "$OUT_DIR"

openssl req -x509 -newkey rsa:4096 -keyout "$OUT_DIR/server.key" -out "$OUT_DIR/server.pem" \
  -days 365 -nodes \
  -subj "/CN=localhost" \
  -addext "subjectAltName=DNS:localhost,DNS:127.0.0.1,IP:127.0.0.1"

chmod 600 "$OUT_DIR/server.key"
echo "Created: $OUT_DIR/server.pem, $OUT_DIR/server.key"
echo ""
echo "Add to settings.json:"
echo "  \"tls_cert_path\": \"$OUT_DIR/server.pem\","
echo "  \"tls_key_path\": \"$OUT_DIR/server.key\""
echo ""
echo "Client: use wss://localhost:9745 and --tls-insecure (or add server.pem as custom CA)"
