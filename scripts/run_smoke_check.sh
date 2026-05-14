#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/smoke_check_literace.py
