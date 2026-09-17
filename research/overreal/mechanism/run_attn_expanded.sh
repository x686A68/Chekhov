#!/bin/bash
# Full-set expanded-prompt attention runs, one chain per GPU.
PY=~/miniconda3/envs/Fraud/bin/python
cd "$(dirname "$0")"
g=$1
run() { CUDA_VISIBLE_DEVICES=$g $PY $1 --cond $2 --shard $3 > logs/attn_x_$4_$2_$(echo $3 | tr / _).log 2>&1; }
case $g in
  0) run attn_flux.py qwen 0/2 flux; run attn_flux.py ideogram 0/2 flux; run attn_qwen.py ideogram 0/3 qwen-image ;;
  1) run attn_flux.py qwen 1/2 flux; run attn_flux.py ideogram 1/2 flux; run attn_qwen.py ideogram 1/3 qwen-image ;;
  2) run attn_sd35.py qwen 0/2 sd35l; run attn_sd35.py ideogram 0/2 sd35l; run attn_qwen.py ideogram 2/3 qwen-image ;;
  3) run attn_sd35.py qwen 1/2 sd35l; run attn_sd35.py ideogram 1/2 sd35l ;;
  4) run attn_qwen.py qwen 0/2 qwen-image ;;
  5) run attn_qwen.py qwen 1/2 qwen-image ;;
esac
