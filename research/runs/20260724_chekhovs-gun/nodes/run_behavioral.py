"""
Behavioral experiment: measure lexical intrusion rate (Chekhov's Gun).
For each paired item, generate the model's answer to Q under treatment (gun present)
and control (gun absent), then detect whether any distractor keyword appears.

Usage:
  python run_behavioral.py --model Qwen/Qwen3-8B --data data_dialogue.json \
     --out behavioral/dlg --limit 0 --max_new_tokens 200
"""
import json, os, re, argparse, time, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load(model_name):
    tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16).to("cuda:0")
    model.eval()
    return tok, model

def build_prompt(tok, messages):
    # Qwen3: disable thinking mode for clean, comparable short answers.
    try:
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

def strip_think(text):
    # Remove any <think>...</think> block if the model still emits one.
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def intrudes(answer, keywords):
    a = answer.lower()
    for k in keywords:
        if re.search(k.lower(), a):
            return True, k
    return False, None

@torch.no_grad()
def generate_batch(tok, model, prompts, max_new_tokens):
    enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    gen = out[:, enc["input_ids"].shape[1]:]
    return [strip_think(tok.decode(g, skip_special_tokens=True)) for g in gen]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    with open(args.data) as f:
        items = json.load(f)
    if args.limit > 0:
        items = items[:args.limit]

    t0 = time.time()
    tok, model = load(args.model)
    print(f"loaded {args.model} in {time.time()-t0:.1f}s", flush=True)

    # Flatten: two conditions per item.
    jobs = []
    for it in items:
        jobs.append((it["id"], "treatment", it))
        jobs.append((it["id"], "control", it))

    results = {}
    for i in range(0, len(jobs), args.batch):
        chunk = jobs[i:i+args.batch]
        prompts = [build_prompt(tok, it[cond]) for (_, cond, it) in chunk]
        answers = generate_batch(tok, model, prompts, args.max_new_tokens)
        for (iid, cond, it), ans in zip(chunk, answers):
            hit, kw = intrudes(ans, it["keywords"])
            results.setdefault(iid, {"id": iid, "task": it["task"],
                                     "distractor_id": it["distractor_id"],
                                     "distance": it["distance"],
                                     "keywords": it["keywords"]})
            results[iid][cond] = {"answer": ans, "intrusion": hit, "kw": kw}
        print(f"  {min(i+args.batch,len(jobs))}/{len(jobs)} done "
              f"({time.time()-t0:.0f}s)", flush=True)

    rows = list(results.values())
    with open(os.path.join(args.out, "results.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Aggregate paired counts.
    t_hit = sum(1 for r in rows if r["treatment"]["intrusion"])
    c_hit = sum(1 for r in rows if r["control"]["intrusion"])
    n = len(rows)
    # McNemar discordant cells
    b = sum(1 for r in rows if r["treatment"]["intrusion"] and not r["control"]["intrusion"])
    c = sum(1 for r in rows if r["control"]["intrusion"] and not r["treatment"]["intrusion"])
    summary = {
        "model": args.model, "data": os.path.basename(args.data), "n_pairs": n,
        "treatment_intrusion_rate": t_hit / n if n else 0,
        "control_intrusion_rate": c_hit / n if n else 0,
        "effect": (t_hit - c_hit) / n if n else 0,
        "mcnemar_b_only_treatment": b, "mcnemar_c_only_control": c,
        "seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("FINAL_METRIC:", summary["effect"])
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
