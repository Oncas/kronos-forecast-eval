#!/usr/bin/env bash
# Runs sequentially so jobs don't fight over CPU cores.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
run() { echo "=== $* ==="; $PY run_forecasts.py "$@" 2>&1 | grep -E "origins|saved|Error|error"; }
run --model small --pred-len 1 --out forecasts_small.npz
run --model mini  --pred-len 1 --out forecasts_mini.npz
run --model base  --pred-len 1 --out forecasts_base.npz
run --model small --pred-len 1 --start 2024-07-01 --end 2025-08-31 --out forecasts_small_ext.npz
echo ALL DONE
