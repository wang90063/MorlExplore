#!/usr/bin/env bash
# Stage 0b sweep: actor type x number of goals x width x seed.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-./.venv/bin/python}
OUT=${OUT:-runs0b}
JOBS=${JOBS:-8}
STEPS=${STEPS:-60000}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

for actor in mlp multi; do
  for K in 3 5; do
    for width in 64 256; do
      for seed in 0 1 2; do
        echo "--actor $actor --K $K --width $width --seed $seed --steps $STEPS --out $OUT"
      done
    done
  done
done | xargs -P "$JOBS" -L 1 "$PY" exp0b.py
