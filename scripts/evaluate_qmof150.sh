#!/usr/bin/env bash
# QMOF150 benchmark (Linux): MOFChecker validity, MOFid novelty and uniqueness, and
# similarity to training structures for 10,000 samples; MOFChecker validity again
# after relaxation.
# Usage: scripts/evaluate_qmof150.sh <flow checkpoint> <output directory>
# Requires: python -m eval.setup mofchecker && python -m eval.setup mofid && python -m eval.setup esen
set -euo pipefail
cd "$(dirname "$0")/.."
checkpoint=${1:?flow checkpoint}
out=${2:?output directory}
workers=${WORKERS:-16}
mofchecker=cache/environments/mofchecker/bin/python

# The training structures are the novelty and similarity reference; prepare them if needed.
root=$(uv run python -m glass.data configs/qmof150.yaml)
training=$root/train.pt

uv run python -m glass.sample --config configs/qmof150.yaml --checkpoint "$checkpoint" --out "$out"

# MOFids of the training structures, the novelty reference (computed once).
if [[ ! -f "$root/train_mofid.csv" ]]; then
  uv run python -m eval.mofid --input "$training" --out "$root/train_mofid.csv" --workers "$workers"
fi

"$mofchecker" -m eval.mofchecker --input "$out/samples.pt" --out "$out/mofchecker.csv" --workers "$workers"
uv run python -m eval.mofid --input "$out/samples.pt" --out "$out/mofid.csv" --workers "$workers"
uv run python -m eval.mof_metrics --samples "$out/samples.pt" --mofchecker "$out/mofchecker.csv" \
  --mofid "$out/mofid.csv" --training-mofid "$root/train_mofid.csv" --training "$training" \
  --out "$out/mof_metrics.json"
uv run python -m eval.training_matches --input "$out/samples.pt" --training "$training" \
  --validity "$out/mofchecker.csv" --out "$out/training_matches.csv" --workers "$workers"

cache/environments/esen/bin/python -m eval.relax --input "$out/samples.pt" --out "$out/relaxed"
"$mofchecker" -m eval.mofchecker --input "$out/relaxed" --out "$out/relaxed_mofchecker.csv" --workers "$workers"
