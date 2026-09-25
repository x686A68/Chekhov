#!/bin/bash
# Orchestrate the remaining intervention runs once the SD3.5-L full run frees the GPUs.
# Chains per GPU; each step is resumable (existing files are skipped).
cd "$(dirname "$0")"
PY=/home/jiahao_huang/miniconda3/envs/Fraud/bin/python
C4="cue_x8,tgt_d8,rand_x8,P_rep_x8"
waitfor() { while pgrep -f "^$PY $1" > /dev/null; do sleep 60; done; }
log() { echo "$(date +%H:%M) $*"; }

# GPU0 / GPU1: SD3.5-L all-rows cue condition, after the six main shards
( waitfor "intervene_sd35.py --shard"; log "sd35 main done; all-rows shards start"
  CUDA_VISIBLE_DEVICES=0 $PY intervene_sd35.py --shard 0/2 --conds cue_x8_all > logs/intervene_sd35l_all_0.log 2>&1 & 
  CUDA_VISIBLE_DEVICES=1 $PY intervene_sd35.py --shard 1/2 --conds cue_x8_all > logs/intervene_sd35l_all_1.log 2>&1 &
  wait; log "sd35 all-rows done"
  CUDA_VISIBLE_DEVICES=0 $PY intervene_judge.py --model sd35l >> logs/judge_sd35l.log 2>&1; log "sd35 judge final done" ) &

# GPU2 / GPU3: FLUX with the fixed partition, then a fill pass
( waitfor "intervene_sd35.py --shard"
  CUDA_VISIBLE_DEVICES=2 $PY intervene_flux.py --shard 0/3 --items intervene_items_small.json --seeds 0 --conds $C4 > logs/intervene_flux_n0.log 2>&1 &
  CUDA_VISIBLE_DEVICES=3 $PY intervene_flux.py --shard 2/3 --items intervene_items_small.json --seeds 0 --conds $C4 > logs/intervene_flux_n2.log 2>&1 &
  wait; waitfor "intervene_flux.py --shard 1/3"
  CUDA_VISIBLE_DEVICES=2 $PY intervene_flux.py --shard 0/1 --items intervene_items_small.json --seeds 0 --conds $C4 > logs/intervene_flux_fill.log 2>&1
  log "flux generation done: $(ls intervene/flux/*/ | grep -c jpg)"
  CUDA_VISIBLE_DEVICES=3 $PY intervene_judge.py --model flux >> logs/judge_flux.log 2>&1; log "flux judge done" ) &

# GPU4 / GPU5: Qwen-Image with the fixed partition, then a fill pass
( waitfor "intervene_flux.py --shard 1/3"
  CUDA_VISIBLE_DEVICES=4 $PY intervene_qwen.py --shard 0/3 --items intervene_items_tiny.json --seeds 0 --conds $C4 > logs/intervene_qwen_n0.log 2>&1 &
  waitfor "intervene_qwen.py --shard 1/3"
  CUDA_VISIBLE_DEVICES=5 $PY intervene_qwen.py --shard 2/3 --items intervene_items_tiny.json --seeds 0 --conds $C4 > logs/intervene_qwen_n2.log 2>&1 &
  wait
  CUDA_VISIBLE_DEVICES=5 $PY intervene_qwen.py --shard 0/1 --items intervene_items_tiny.json --seeds 0 --conds $C4 > logs/intervene_qwen_fill.log 2>&1
  log "qwen generation done: $(ls intervene/qwen-image/*/ | grep -c jpg)"
  CUDA_VISIBLE_DEVICES=4 $PY intervene_judge.py --model qwen-image >> logs/judge_qwen.log 2>&1; log "qwen judge done" ) &

wait
log "ALL DONE"
