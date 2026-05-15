#!/bin/bash
# Standalone pipeline runner for local testing (no submission).
# For production on the Ubuntu VM, call run_daily_submissions.sh in
# energy-arena-participate-working instead.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
python pipeline.py --date "$(date -d 'tomorrow' '+%Y-%m-%d')" "$@"
