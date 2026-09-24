#!/bin/bash
# Judge the wild images (one seed per prompt) with the claude CLI backend, N shards in parallel.
# Requires a logged-in `claude` CLI (run `claude login` first). Resumable per shard.
# Usage: bash launch_judge_cli.sh [N=6] | bash launch_judge_cli.sh stop
cd "$(dirname "$0")/../autoannot" || exit 1
PY=/home/jiahao_huang/miniconda3/envs/Fraud/bin/python
OUT=/home/jiahao_huang/Chekhov/data/overreal_v1/auto
LOG=/home/jiahao_huang/Chekhov/data/generation/logs
if [ "$1" = "stop" ]; then pkill -f "run_auto.py --backend claude" && echo stopped || echo "nothing running"; exit 0; fi
N=${1:-6}
for k in $(seq 0 $((N-1))); do
  nohup "$PY" run_auto.py --backend claude --model claude-opus-5 --protocol final \
    --manifests '*_wild.jsonl' --id-regex '__s0\.png$' --shard $k/$N \
    --out "$OUT/final__claude-opus-5-cli__wild_s$k.jsonl" > "$LOG/judge_cli_wild_s$k.log" 2>&1 &
  echo "judge shard $k/$N pid $!"
done
