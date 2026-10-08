#!/bin/bash
# 5 preferences x 2 seeds; runs with the same seed share their initialisation.
cd "$(dirname "$0")/.."
for s in 0 1; do for c in 0.05 0.2 0.5 1.0 2.0; do echo "$c $s"; done; done |
  xargs -P 8 -n 2 sh -c './.venv/bin/python -W ignore geom/sac_mo.py --c $0 --seed $1 --steps 300000 > geom/logs/c$0_s$1.log 2>&1'
