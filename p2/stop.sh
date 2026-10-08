#!/bin/zsh
# Stop the P2 queue and every training job. Safe at any moment: checkpoints are written to a temp file and
# renamed, so a kill mid-save keeps the previous one. Resume with: caffeinate -i p2/queue.sh 8 1000000
pkill -f "p2/queue.sh"
pkill -f "xargs -P .* -L 1 zsh"
pkill -f "p2/esac.py"
pkill -f "cola_compat/run_cola.py"
sleep 2
left=$(pgrep -f "p2/queue.sh|p2/esac.py|cola_compat/run_cola.py" | wc -l | tr -d ' ')
echo "stopped; processes left: $left"
cd "$(dirname "$0")/.."
ls -lh ${OUT:-runs/p2/kill}/*/ckpt.pt(N)
