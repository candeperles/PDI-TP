#!/usr/bin/env bash
# Compatibilidad con el comando original; la descarga se implementa en Python.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python3}" "$SCRIPT_DIR/download_data.py"
