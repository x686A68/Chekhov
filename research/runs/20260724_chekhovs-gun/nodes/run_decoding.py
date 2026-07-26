"""
Decoding-intervention experiment: does task ENTROPY gate expression?

The task-entropy account predicts that flattening the output distribution
(higher temperature) should let the (constant) availability boost surface more
often, and sharpening it (greedy / low temp) should suppress it. We therefore
sweep sampling temperature and measure the behavioral intrusion rate on the
same paired stimuli, with multiple samples per item for stable estimates.

Prediction: intrusion rate increases monotonically with temperature; control
stays ~0 at all temperatures.

Usage:
  python run_decoding.py --data data_dialogue_gen.json --out decoding/gen \
     --temps 0.0 0.7 1.0 1.3 --samples 4
"""
import json, os, re, argparse, time
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
    try:
        return tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def strip_think(t):
    return re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL).strip()

def intrudes(ans, keywords):
    a = ans.lower()
    return any(re.search(k.lower(), a) for k in keywords)

@torch.no_grad()
def gen_batch(tok, model, prompts, temp, top_p, max_new, seed):
    enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    torch.manual_seed(seed)
    kw = dict(max_new_tokens=max_new, pad_token_id=tok.pad_token_id)
    if temp <= 0:
        kw.update(do_sample=False)
    else:
        kw.update(do_sample=True, temperature=temp, top_p=top_p)
    out = model.generate(**enc, **kw)
    gen = out[:, enc["input_ids"].shape[1]:]
    return [strip_think(tok.decode(g, skip_special_tokens=True)) for g in gen]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--temps", type=float, nargs="+", default=[0.0, 0.7, 1.0, 1.3])
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=160)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    with open(args.data) as f:
        items = json.load(f)
    if args.limit > 0:
        items = items[:args.limit]

    t0 = time.time()
    tok, model = load(args.model)
    print(f"loaded {args.model} in {time.time()-t0:.1f}s", flush=True)

    curve = []
    for temp in args.temps:
        n_samp = 1 if temp <= 0 else args.samples
        # accumulate per-condition intrusion counts over samples
        t_hits = c_hits = total = 0
        for s in range(n_samp):
            jobs = []
            for it in items:
                jobs.append((it, "treatment"))
                jobs.append((it, "control"))
            for i in range(0, len(jobs), args.batch):
                chunk = jobs[i:i+args.batch]
                prompts = [build_prompt(tok, it[cond]) for it, cond in chunk]
                ans = gen_batch(tok, model, prompts, temp, args.top_p,
                                args.max_new_tokens, seed=1000*s + i)
                for (it, cond), a in zip(chunk, ans):
                    hit = intrudes(a, it["keywords"])
                    if cond == "treatment":
                        t_hits += hit
                    else:
                        c_hits += hit
                    if cond == "treatment":
                        total += 1
        rate_t = t_hits / total
        rate_c = c_hits / total
        curve.append({"temperature": temp, "samples": n_samp,
                      "treatment_intrusion_rate": rate_t,
                      "control_intrusion_rate": rate_c,
                      "n_item_samples": total})
        print(f"  T={temp}: treat={rate_t:.3f} ctrl={rate_c:.3f} "
              f"(n={total}, {time.time()-t0:.0f}s)", flush=True)

    summary = {"model": args.model, "data": os.path.basename(args.data),
               "top_p": args.top_p, "curve": curve,
               "seconds": round(time.time() - t0, 1)}
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("FINAL_METRIC:", curve[-1]["treatment_intrusion_rate"])
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
