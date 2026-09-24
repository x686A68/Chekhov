#!/bin/bash
# Launch (or resume) the five open-model runs on the wild prompt set, one GPU each.
# Resumable: run_local.py skips (item, seed) pairs already in the manifest.
# Usage: bash research/overreal/scripts/wild/launch_local_wild.sh [stop]
cd "$(dirname "$0")/../gen" || exit 1
PY=/home/jiahao_huang/miniconda3/envs/Fraud/bin/python
LOG=../../../../data/generation/logs
mkdir -p "$LOG"
if [ "$1" = "stop" ]; then
  pkill -f "run_local.py --model" && echo "stopped" || echo "nothing running"
  exit 0
fi
i=0
for m in flux qwen-image omnigen2 sd35m sd35l; do
  CUDA_VISIBLE_DEVICES=$i nohup "$PY" run_local.py --model "$m" --cond wild >> "$LOG/wild_$m.log" 2>&1 &
  echo "started $m on GPU $i (pid $!)"
  i=$((i+1))
done
