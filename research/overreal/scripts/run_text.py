"""Phase 0 — run the text pilot.

Greedy decoding, one model at a time, all 252 prompts per model. Raw generations are
written to pilot/text/raw/<model>.jsonl as they are produced; scoring happens
separately in score_text.py so the scorer can be revised without regenerating.

Usage: CUDA_VISIBLE_DEVICES=4,5 python scripts/run_text.py <model_key> [...]
"""
import json
import os
import sys
import time

os.environ.setdefault("HF_HOME", "/data/users/jiahao_huang/hf")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
# the installed vllm warms up FP8 DeepGEMM kernels that the env has no deep_gemm for
os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ITEMS = os.path.join(ROOT, "pilot", "items")
RAW = os.path.join(ROOT, "pilot", "text", "raw")

MODELS = {
    "qwen3-8b": dict(path="Qwen/Qwen3-8B", tp=1, thinking=False),
    "qwen3-32b": dict(path="Qwen/Qwen3-32B", tp=1, thinking=False),
    "llama3.1-8b": dict(path="meta-llama/Llama-3.1-8B-Instruct", tp=1, thinking=None),
}

MAX_TOKENS = 300


def load_items():
    rows = []
    for fn in sorted(os.listdir(ITEMS)):
        if fn.endswith(".jsonl"):
            with open(os.path.join(ITEMS, fn)) as f:
                rows += [json.loads(l) for l in f if l.strip()]
    return rows


def main(keys):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    os.makedirs(RAW, exist_ok=True)
    items = load_items()
    print(f"{len(items)} items, {len(items)*3} prompts per model", flush=True)

    for key in keys:
        cfg = MODELS[key]
        out_path = os.path.join(RAW, f"{key}.jsonl")
        tok = AutoTokenizer.from_pretrained(cfg["path"])

        flat = []  # (item, condition, prompt_text)
        for it in items:
            for cond in ("S", "P", "A"):
                kw = {} if cfg["thinking"] is None else {"enable_thinking": cfg["thinking"]}
                text = tok.apply_chat_template(
                    it["prompts"][cond], tokenize=False, add_generation_prompt=True, **kw
                )
                flat.append((it, cond, text))

        t0 = time.time()
        llm = LLM(model=cfg["path"], tensor_parallel_size=cfg["tp"], dtype="bfloat16",
                  gpu_memory_utilization=0.85, max_model_len=2048, enforce_eager=False,
                  disable_log_stats=True)
        load_s = time.time() - t0

        t1 = time.time()
        outs = llm.generate([t for _, _, t in flat],
                            SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS))
        gen_s = time.time() - t1

        with open(out_path, "w") as f:
            for (it, cond, text), o in zip(flat, outs):
                f.write(json.dumps({
                    "model": key, "id": it["id"], "family": it["family"],
                    "entity": it["entity"], "scenario_id": it["scenario_id"],
                    "scenario": it["scenario"], "device": it["device"], "condition": cond,
                    "prompt": it["prompts"][cond], "prompt_text": text,
                    "output": o.outputs[0].text.strip(),
                    "n_out_tokens": len(o.outputs[0].token_ids),
                }, ensure_ascii=False) + "\n")

        cost = {"model": key, "n_prompts": len(flat), "load_s": round(load_s, 1),
                "gen_s": round(gen_s, 1), "s_per_prompt": round(gen_s / len(flat), 3),
                "max_tokens": MAX_TOKENS, "tp": cfg["tp"], "gpu_mem_util": 0.85}
        with open(os.path.join(RAW, f"{key}.cost.json"), "w") as f:
            json.dump(cost, f, indent=2)
        print("COST", json.dumps(cost), flush=True)

        del llm
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main(sys.argv[1:] or ["qwen3-8b"])
