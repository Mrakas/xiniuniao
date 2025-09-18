#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

PYTHON_BIN=${PYTHON_BIN:-python}

GPU0=${1:-0}
GPU1=${2:-1}
GPU2=${3:-2}

pids=()

echo "Starting run_flux_1.py on GPU ${GPU0}"
CUDA_VISIBLE_DEVICES="${GPU0}" "$PYTHON_BIN" run_flux_1.py &
pids+=($!)

echo "Starting run_flux_2.py on GPU ${GPU1}"
CUDA_VISIBLE_DEVICES="${GPU1}" "$PYTHON_BIN" run_flux_2.py &
pids+=($!)

echo "Starting run_flux_3.py on GPU ${GPU2}"
CUDA_VISIBLE_DEVICES="${GPU2}" "$PYTHON_BIN" run_flux_3.py &
pids+=($!)

cleanup() {
  trap - SIGINT SIGTERM
  if [ ${#pids[@]} -gt 0 ]; then
    kill "${pids[@]}" 2>/dev/null || true
  fi
}

trap cleanup SIGINT SIGTERM

wait "${pids[@]}"
