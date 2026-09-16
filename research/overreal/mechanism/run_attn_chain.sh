#!/bin/bash
# Per GPU: wait for the FLUX shard to finish, then run the SD3.5-L and Qwen-Image shards.
g=$1; n=$2
PY=~/miniconda3/envs/Fraud/bin/python
cd "$(dirname "$0")"
while pgrep -f "attn_flux.py --shard $g/$n" > /dev/null; do sleep 60; done
CUDA_VISIBLE_DEVICES=$g $PY attn_sd35.py --shard $g/$n > logs/attn_sd35l_$g.log 2>&1
CUDA_VISIBLE_DEVICES=$g $PY attn_qwen.py --shard $g/$n > logs/attn_qwen_$g.log 2>&1
