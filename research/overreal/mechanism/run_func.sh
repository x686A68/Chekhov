#!/bin/bash
cd "$(dirname "$0")"
PY=/home/jiahao_huang/miniconda3/envs/Fraud/bin/python
until grep -q "sd35 all-rows done" logs/orchestrator.log; do sleep 120; done
CUDA_VISIBLE_DEVICES=0 $PY intervene_sd35.py --shard 0/2 --seeds 0 --conds func_x8 > logs/intervene_sd35l_func_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY intervene_sd35.py --shard 1/2 --seeds 0 --conds func_x8 > logs/intervene_sd35l_func_1.log 2>&1 &
wait
CUDA_VISIBLE_DEVICES=1 $PY intervene_judge.py --model sd35l >> logs/judge_sd35l.log 2>&1
echo "$(date +%H:%M) func_x8 done and judged"
