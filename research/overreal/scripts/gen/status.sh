#!/bin/bash
# One-glance progress for all generation runs.
cd ~/Chekhov/data/generation
for f in manifests/*.jsonl; do
  n=$(wc -l < "$f")
  case "$(basename $f .jsonl)" in
    ideogram_deployed|nanobanana_deployed) tot=548;;
    *_deployed) tot=1096;;
    *_raw) tot=1096;;
    *) tot=?;;
  esac
  printf "%-24s %s/%s\n" "$(basename $f .jsonl)" "$n" "$tot"
done
echo "expanded prompts: $(sort -u -t'"' -k8,8 expanded_prompts.jsonl 2>/dev/null | wc -l) lines"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
