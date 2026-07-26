"""
Semantic intrusion detection (tightens the behavioral lower bound).

Keyword matching only catches verbatim topical carry-over. Here we measure
paraphrastic/thematic intrusion: for each paired item we embed the treatment
answer, the control answer, and a canonical phrase for the distractor topic
(built from the item's keywords), using BAAI/bge-large-en-v1.5, and compute
cosine similarity of each answer to the topic.

Paired semantic-intrusion signal:
  delta_sim = cos(treatment_answer, topic) - cos(control_answer, topic)
delta_sim > 0 means the treatment answer is drawn semantically closer to the
distractor topic than the control answer -- intrusion even without the keyword.

We report mean delta_sim, paired bootstrap 95% CI, and the fraction of pairs
with positive delta. This complements the keyword rate, especially in the
scoped/reading regimes where keyword intrusion is ~0.

Usage:
  python run_semantic.py --results behavioral/full_gen/results.jsonl --out semantic/gen
"""
import json, os, re, argparse, time
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

MODEL = "BAAI/bge-large-en-v1.5"

def load():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, dtype=torch.float16).to("cuda:0")
    model.eval()
    return tok, model

@torch.no_grad()
def embed(tok, model, texts, batch=32):
    outs = []
    for i in range(0, len(texts), batch):
        chunk = [t if t.strip() else "." for t in texts[i:i+batch]]
        enc = tok(chunk, padding=True, truncation=True, max_length=256,
                  return_tensors="pt").to(model.device)
        out = model(**enc)
        emb = out.last_hidden_state[:, 0]           # CLS pooling (bge convention)
        emb = F.normalize(emb, p=2, dim=1)
        outs.append(emb.float().cpu())
    return torch.cat(outs, 0)

def topic_phrase(keywords):
    parts = []
    for k in keywords:
        w = re.sub(r"[^a-z ]", "", k.lower()).strip()
        if w:
            parts.append(w)
    return " ".join(dict.fromkeys(parts)) or "unknown topic"

def boot_ci(vals, B=2000):
    import random
    random.seed(0)
    n = len(vals); means = []
    for _ in range(B):
        s = [vals[random.randrange(n)] for _ in range(n)]
        means.append(sum(s) / n)
    means.sort()
    return means[int(0.025 * B)], means[int(0.975 * B)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = [json.loads(l) for l in open(args.results)]

    t0 = time.time()
    tok, model = load()
    print(f"loaded bge in {time.time()-t0:.1f}s", flush=True)

    treat_ans = [r["treatment"]["answer"] for r in rows]
    ctrl_ans = [r["control"]["answer"] for r in rows]
    topics = [topic_phrase(r["keywords"]) for r in rows]

    et = embed(tok, model, treat_ans)
    ec = embed(tok, model, ctrl_ans)
    eo = embed(tok, model, topics)

    out_rows, deltas = [], []
    for i, r in enumerate(rows):
        ct = float((et[i] * eo[i]).sum())
        cc = float((ec[i] * eo[i]).sum())
        d = ct - cc
        deltas.append(d)
        out_rows.append({"id": r["id"], "task": r["task"], "distance": r["distance"],
                         "cos_treat_topic": ct, "cos_ctrl_topic": cc, "delta_sim": d})

    with open(os.path.join(args.out, "results.jsonl"), "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")

    import statistics as st
    lo, hi = boot_ci(deltas)
    summary = {
        "results": args.results, "n": len(rows),
        "mean_cos_treat_topic": st.mean(r["cos_treat_topic"] for r in out_rows),
        "mean_cos_ctrl_topic": st.mean(r["cos_ctrl_topic"] for r in out_rows),
        "mean_delta_sim": st.mean(deltas),
        "delta_sim_ci95": [lo, hi],
        "frac_pairs_positive": sum(1 for d in deltas if d > 0) / len(deltas),
        "seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("FINAL_METRIC:", summary["mean_delta_sim"])
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
