"""
Mechanistic experiment (teacher-forced continuation surprisal).

The behavioral study shows the distractor surfaces only in generative regimes.
Here we ask the sharper question: is the distractor concept PRIMED (more available)
even when it does not surface? We measure it as the log-probability the model assigns
to actually SAYING the distractor keyword at a content-generation point, holding
everything identical except the presence of the "gun".

For each paired item:
  prefix = chat_prompt(context)  +  a FIXED, shared neutral answer opener
           (e.g. "Sure! Here's something that comes to mind: ")
  For each surface realization s of the distractor keyword, compute the summed
  token log-prob of s as the continuation of prefix (teacher forcing).
  score = max over realizations (the model's best way of starting to say it).

  delta_logp = score_treatment - score_control   (>0 => primed)

Because the opener is identical across treatment and control, delta_logp isolates
the causal effect of the planted distractor on the availability of its own topic.

Usage: python run_mechanistic.py --data data_dialogue.json --out mech/scoped
"""
import json, os, re, argparse, time
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

OPENER = "Sure! Here's something that comes to mind: "

def load(model_name):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16).to("cuda:0")
    model.eval()
    return tok, model

def prompt_text(tok, messages):
    try:
        return tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def realizations(keywords):
    """Natural surface forms of the distractor topic to score."""
    outs = []
    for k in keywords:
        w = re.sub(r"[^a-z ]", "", k.lower()).strip()
        if not w:
            continue
        outs.append(" " + w)
        outs.append(" " + w.capitalize())
        outs.append(" " + w.title())
    return list(dict.fromkeys(outs))

@torch.no_grad()
def continuation_logprob(tok, model, prefix_text, continuation):
    """Sum log P(continuation tokens | prefix) under teacher forcing."""
    pre = tok(prefix_text, add_special_tokens=False)["input_ids"]
    con = tok(continuation, add_special_tokens=False)["input_ids"]
    if not con:
        return None
    ids = torch.tensor([pre + con], device=model.device)
    logits = model(ids).logits[0]                 # [T, V]
    logp = F.log_softmax(logits.float(), dim=-1)
    total = 0.0
    for j, tid in enumerate(con):
        pos = len(pre) + j - 1                     # distribution predicting token at len(pre)+j
        total += logp[pos, tid].item()
    return total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    with open(args.data) as f:
        items = json.load(f)
    if args.limit > 0:
        items = items[:args.limit]

    t0 = time.time()
    tok, model = load(args.model)
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    rows = []
    for j, it in enumerate(items):
        reals = realizations(it["keywords"])
        pre_t = prompt_text(tok, it["treatment"]) + OPENER
        pre_c = prompt_text(tok, it["control"]) + OPENER
        # best (max) continuation logprob over surface realizations
        st = max(continuation_logprob(tok, model, pre_t, r) for r in reals)
        sc = max(continuation_logprob(tok, model, pre_c, r) for r in reals)
        rows.append({"id": it["id"], "task": it["task"], "distance": it["distance"],
                     "distractor_id": it["distractor_id"],
                     "logp_treatment": st, "logp_control": sc, "delta_logp": st - sc})
        if (j + 1) % 20 == 0:
            print(f"  {j+1}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)

    with open(os.path.join(args.out, "results.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    import statistics as st_
    deltas = [r["delta_logp"] for r in rows]
    # paired bootstrap 95% CI on mean delta
    def boot_ci(vals, B=2000):
        import random
        random.seed(0)
        n = len(vals); means = []
        for _ in range(B):
            s = [vals[random.randrange(n)] for _ in range(n)]
            means.append(sum(s)/n)
        means.sort()
        return means[int(0.025*B)], means[int(0.975*B)]
    lo, hi = boot_ci(deltas)
    by_dist = {}
    for r in rows:
        by_dist.setdefault(r["distance"], []).append(r["delta_logp"])
    summary = {
        "model": args.model, "data": os.path.basename(args.data), "n": len(rows),
        "mean_logp_treatment": st_.mean(r["logp_treatment"] for r in rows),
        "mean_logp_control": st_.mean(r["logp_control"] for r in rows),
        "mean_delta_logp": st_.mean(deltas),
        "delta_logp_ci95": [lo, hi],
        "frac_pairs_primed": sum(1 for d in deltas if d > 0) / len(deltas),
        "median_delta_logp": st_.median(deltas),
        "delta_by_distance": {d: st_.mean(v) for d, v in sorted(by_dist.items())},
        "seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("FINAL_METRIC:", summary["mean_delta_logp"])
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
