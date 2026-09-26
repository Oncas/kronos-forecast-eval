#!/usr/bin/env bash
# Second pass with the paper's daily-frequency lookback (40 bars). The 400-bar
# runs normalised over a long trend, which made token resolution coarse and
# the forecasts mostly "revert to the window mean".
set -euo pipefail
cd "$(dirname "$0")"
PY=../../.venv/bin/python
EXT="--start 2024-07-01 --end 2025-08-31"
run() { echo "=== $* ==="; $PY run_forecasts.py "$@" 2>&1 | grep --line-buffered -E "origins|saved|Error|error"; }
for m in small mini base; do
  run --model $m --lookback 40 --pred-len 1 --out forecasts_${m}_lb40.npz
  run --model $m --lookback 40 --pred-len 1 $EXT --out forecasts_${m}_ext_lb40.npz
done
run --model small --lookback 40 --pred-len 1 --top-p 1.0 --out forecasts_small_lb40_topp1.npz
run --model small --lookback 90 --pred-len 1 --out forecasts_small_lb90.npz
run --model small --lookback 90 --pred-len 1 $EXT --out forecasts_small_ext_lb90.npz
run --model small --lookback 40 --pred-len 7 --samples 20 --out forecasts_small_7d_lb40.npz
echo ALL DONE
