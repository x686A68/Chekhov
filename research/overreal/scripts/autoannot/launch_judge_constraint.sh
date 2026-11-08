#!/bin/bash
# Judge the explicit-constraint images (KXAW Q2) with the Anthropic API backend, N shards in parallel.
# Usage: bash launch_judge_constraint.sh [N=5] | bash launch_judge_constraint.sh stop
cd "$(dirname "$0")" || exit 1
PY=/home/jiahao_huang/miniconda3/envs/Fraud/bin/python
OUT=/data/users/jiahao_huang/Chekhov/data/overreal_v1/auto
LOG=/data/users/jiahao_huang/Chekhov/research/overreal/logs
if [ "$1" = "stop" ]; then pkill -f "run_auto.py --backend anthropic.*_constraint" && echo stopped || echo "nothing running"; exit 0; fi
N=${1:-5}
for k in $(seq 0 $((N-1))); do
  nohup "$PY" run_auto.py --backend anthropic --model claude-opus-5 --protocol final \
    --manifests '*_constraint.jsonl' --shard $k/$N --max-usd 20 \
    --out "$OUT/final__claude-opus-5-api__constraint_s$k.jsonl" > "$LOG/judge_constraint_s$k.log" 2>&1 &
  echo "judge shard $k/$N pid $!"
done
