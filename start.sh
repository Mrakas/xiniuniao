#!/usr/bin/env bash
# Launch the three Flux runners on separate machines with explicit CUDA device binding.
set -euo pipefail

ROOT_DIR="/mnt/sh/mmvision/home/marcuskwan/ghx/xiniiuniao"
SCRIPTS=(
  "run_flux_1.py"
  "run_flux_2.py"
  "run_flux_3.py"
)
# TODO: Replace these placeholders with the actual hostnames or SSH targets.
HOSTS=(
  "machine-a"
  "machine-b"
  "machine-c"
)
CUDA_VISIBLE=(
  "0"
  "0"
  "0"
)

if [[ ${#SCRIPTS[@]} -ne 3 || ${#HOSTS[@]} -ne 3 || ${#CUDA_VISIBLE[@]} -ne 3 ]]; then
  echo "The SCRIPTS, HOSTS, and CUDA_VISIBLE arrays must each contain exactly three entries." >&2
  exit 1
fi

for idx in "${!SCRIPTS[@]}"; do
  host="${HOSTS[$idx]}"
  script="${SCRIPTS[$idx]}"
  cuda_id="${CUDA_VISIBLE[$idx]}"

  echo "Launching ${script} on ${host} with CUDA_VISIBLE_DEVICES=${cuda_id}"
  ssh "${host}" "cd \"${ROOT_DIR}\" && mkdir -p logs && CUDA_VISIBLE_DEVICES=${cuda_id} nohup python3 \"${ROOT_DIR}/${script}\" > logs/${script%.py}.log 2>&1 &" &
done

wait

echo "All launch commands have been issued."
