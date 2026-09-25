#!/bin/bash
cd "$(dirname "$0")"
PY=/home/jiahao_huang/miniconda3/envs/Fraud/bin/python
CUDA_VISIBLE_DEVICES=2 $PY intervene_flux.py --shard 0/2 --items intervene_items_small.json --seeds 0 --conds P_tgt_d8 > logs/intervene_flux_ptgt_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 $PY intervene_flux.py --shard 1/2 --items intervene_items_small.json --seeds 0 --conds P_tgt_d8 > logs/intervene_flux_ptgt_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 $PY intervene_qwen.py --shard 0/2 --items intervene_items_tiny.json --seeds 0 --conds P_tgt_d8 > logs/intervene_qwen_ptgt_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 $PY intervene_qwen.py --shard 1/2 --items intervene_items_tiny.json --seeds 0 --conds P_tgt_d8 > logs/intervene_qwen_ptgt_1.log 2>&1 &
wait
CUDA_VISIBLE_DEVICES=2 $PY intervene_judge.py --model flux >> logs/judge_flux.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 $PY intervene_judge.py --model qwen-image >> logs/judge_qwen.log 2>&1 &
wait
echo "$(date +%H:%M) P_tgt_d8 done and judged"
