#!/usr/bin/env bash
# Fixed-grid toy: train and sample every (N, seed, method) run of the paper.
# Usage: scripts/toy.sh                 # full study, sequentially
#        scripts/toy.sh <N> <seed> <method>   # one run; method is naive, eqot, or decoder
# Runs are independent; to parallelize, launch single runs on separate GPUs.
set -euo pipefail
cd "$(dirname "$0")/.."

run() {
  local out=runs/toy/n$1_seed$2_$3
  uv run python -m toy.fixed_grid train --n "$1" --seed "$2" --method "$3" --out "$out"
  uv run python -m toy.fixed_grid sample --run "$out"
}

if [[ $# -eq 3 ]]; then
  run "$@"
  exit
fi

for n in 4 8 12 16 20 24 28 32 40 48 64; do
  for seed in 0 1 2; do
    for method in naive eqot decoder; do
      run "$n" "$seed" "$method"
    done
  done
done
