#!/usr/bin/env bash
# Stage 0a sweep: mode x width x objective correlation x seed, run in parallel on CPU.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-./.venv/bin/python}
OUT=${OUT:-runs}
JOBS=${JOBS:-8}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

for mode in sup fqi; do
  for width in 16 64 256; do
    for corr in 0.0 0.7; do
      for seed in 0 1 2 3 4; do
        echo "--mode $mode --width $width --corr $corr --seed $seed --out $OUT"
      done
    done
  done
done | xargs -P "$JOBS" -L 1 "$PY" exp0a.py
