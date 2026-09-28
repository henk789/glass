#!/usr/bin/env bash
# MP20 benchmark (Linux, CUDA): LeMat-GenBench on 2,500 raw and 2,500 NequIP-relaxed
# samples, and Crystalite's DNG metrics on 10,000 samples.
# Usage: scripts/evaluate_mp20.sh <flow checkpoint> <output directory>
# Requires: python -m eval.setup genbench && python -m eval.setup crystalite
set -euo pipefail
cd "$(dirname "$0")/.."
checkpoint=${1:?flow checkpoint}
out=${2:?output directory}
crystalite=cache/upstream/crystalite/.venv/bin/python

uv run python -m glass.sample --config configs/mp20.yaml --checkpoint "$checkpoint" --out "$out"
# LeMat-GenBench scores the first 2,500 of the 10,000 samples.
uv run python -c "
import torch; d = torch.load('$out/samples.pt')
torch.save({k: v[:2500] for k, v in d.items()}, '$out/samples_2500.pt')"

uv run python -m eval.genbench --mode prod --input "$out/samples_2500.pt" --out "$out/lemat_raw.json" --expected-count 2500 --workers 8
"$crystalite" -m eval.nequip_relax --input "$out/samples_2500.pt" --out "$out/nequip_relaxed"
uv run python -m eval.genbench --mode prod --input "$out/nequip_relaxed" --out "$out/lemat_nequip.json" --expected-count 2500 --workers 8
"$crystalite" -m eval.crystalite --input "$out/samples.pt" --out "$out/crystalite.json"
