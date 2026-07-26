"""
Analyze the openness ladder: relate task openness to intrusion rate.

Openness of a task is quantified INDEPENDENTLY of the distractor as the semantic
diversity of the CONTROL-group answers at that level: an open task ("write a
caption") admits many different good answers (high diversity); a constrained task
("when does construction begin?") admits essentially one (low diversity). We embed
control answers with bge-large and take mean pairwise cosine distance as the
openness proxy, then plot intrusion rate against it.
"""
import json, os, itertools
import torch, torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

BASE = os.path.dirname(os.path.abspath(__file__))
LEVELS = ["L0_extractive", "L1_summary", "L2_takeaway", "L3_creative"]
MODEL = "BAAI/bge-large-en-v1.5"

def load():
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModel.from_pretrained(MODEL, dtype=torch.float16).to("cuda:0"); m.eval()
    return tok, m

@torch.no_grad()
def embed(tok, m, texts, batch=32):
    outs = []
    for i in range(0, len(texts), batch):
        chunk = [t if t.strip() else "." for t in texts[i:i+batch]]
        enc = tok(chunk, padding=True, truncation=True, max_length=256,
                  return_tensors="pt").to(m.device)
        e = m(**enc).last_hidden_state[:, 0]
        outs.append(F.normalize(e, p=2, dim=1).float().cpu())
    return torch.cat(outs, 0)

def mean_pairwise_dist(emb, cap=400):
    n = min(len(emb), cap)
    e = emb[:n]
    sim = e @ e.T
    iu = torch.triu_indices(n, n, offset=1)
    d = 1 - sim[iu[0], iu[1]]
    return float(d.mean())

def main():
    tok, m = load()
    rows = []
    for L in LEVELS:
        p = os.path.join(BASE, "ladder", L, "results.jsonl")
        rs = [json.loads(l) for l in open(p)]
        intr = sum(1 for r in rs if r["treatment"]["intrusion"]) / len(rs)
        ctrl = embed(tok, m, [r["control"]["answer"] for r in rs])
        openness = mean_pairwise_dist(ctrl)
        rows.append({"level": L, "intrusion": intr, "openness": openness, "n": len(rs)})
        print(f"{L}: intrusion={intr:.3f} openness={openness:.3f}")

    # Spearman correlation (rank-based, no scipy dependency)
    def spearman(xs, ys):
        def rank(v):
            order = sorted(range(len(v)), key=lambda i: v[i])
            r = [0]*len(v)
            for rr, i in enumerate(order):
                r[i] = rr
            return r
        rx, ry = rank(xs), rank(ys)
        n = len(xs)
        d2 = sum((a-b)**2 for a, b in zip(rx, ry))
        return 1 - 6*d2/(n*(n*n-1))
    xs = [r["openness"] for r in rows]; ys = [r["intrusion"] for r in rows]
    rho = spearman(xs, ys)
    out = {"levels": rows, "spearman_rho": rho}
    with open(os.path.join(BASE, "ladder", "summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("spearman rho:", rho)

    # Figure
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    names = ["extractive", "summary", "takeaway", "creative"]
    ax.plot(xs, ys, "-o", color="#c1432c")
    for x, y, nm in zip(xs, ys, names):
        ax.annotate(nm, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Task openness  (control-answer semantic diversity)")
    ax.set_ylabel("Distractor intrusion rate")
    ax.set_title(f"Intrusion rises monotonically with task openness (rho={rho:.2f})")
    ax.set_ylim(-0.03, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(BASE, "..", "figures", "fig7_openness_ladder.png"), dpi=160)
    print("wrote fig7_openness_ladder.png")

if __name__ == "__main__":
    main()
