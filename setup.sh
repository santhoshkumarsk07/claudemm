#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
command -v "$PY" >/dev/null || { echo "Python 3.11+ not found. Install it from https://www.python.org/downloads/"; exit 1; }
$PY -c 'import sys; assert sys.version_info >= (3, 11), "Python 3.11+ is required"'
$PY -m venv .venv
.venv/bin/pip install --upgrade pip -q
.venv/bin/pip install --require-hashes -r requirements.lock -q
[ -f .env ] || cp .env.example .env
chmod 600 .env
echo
echo "Setup done. Next:"
echo "  source .venv/bin/activate"
echo "  python -m arena demo"
