#!/bin/bash
# Re-run the presence judge for a model every 10 min while its generation runs, then once more.
# usage: judge_loop.sh <model_key> <gpu> <script name to wait for, e.g. intervene_sd35.py>
m=$1; g=$2; pat=$3
PY=~/miniconda3/envs/Fraud/bin/python
cd "$(dirname "$0")"
while ps -eo args | grep -v grep | grep -q "python .*$pat"; do
  CUDA_VISIBLE_DEVICES=$g $PY intervene_judge.py --model $m >> logs/judge_$m.log 2>&1
  sleep 600
done
CUDA_VISIBLE_DEVICES=$g $PY intervene_judge.py --model $m >> logs/judge_$m.log 2>&1
echo "judge loop done for $m"
