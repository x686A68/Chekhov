"""
Aggregate all behavioral + mechanistic results and render paper figures.
Outputs PNGs into ../figures/ and a consolidated results_table.json.
"""
import json, os, glob, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(BASE, "..", "figures")
os.makedirs(FIG, exist_ok=True)

def load_jsonl(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []

def load_summary(p):
    return json.load(open(p)) if os.path.exists(p) else None

def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0, 0)
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (max(0, c-h), min(1, c+h))

# ---- Collect behavioral conditions ----
COND = [("full_scoped", "Dialogue\n(scoped)"),
        ("full_gen", "Dialogue\n(generative)"),
        ("full_reading", "Reading\n(extractive)"),
        ("full_reading_gen", "Reading\n(generative)")]
beh = {}
for key, label in COND:
    s = load_summary(os.path.join(BASE, "behavioral", key, "summary.json"))
    rows = load_jsonl(os.path.join(BASE, "behavioral", key, "results.jsonl"))
    if s:
        beh[key] = {"label": label, "summary": s, "rows": rows}

# ---- Figure 1: main intrusion-rate bar chart with Wilson CIs ----
if beh:
    labels, tr, cr, tr_ci, cr_ci = [], [], [], [], []
    for key, label in COND:
        if key not in beh:
            continue
        s = beh[key]["summary"]; n = s["n_pairs"]
        t = s["treatment_intrusion_rate"]; c = s["control_intrusion_rate"]
        labels.append(beh[key]["label"]); tr.append(t); cr.append(c)
        tr_ci.append(wilson_ci(round(t*n), n)); cr_ci.append(wilson_ci(round(c*n), n))
    import numpy as np
    x = np.arange(len(labels)); w = 0.36
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    def err(vals, cis):
        lo = [v - c[0] for v, c in zip(vals, cis)]
        hi = [c[1] - v for v, c in zip(vals, cis)]
        return [lo, hi]
    ax.bar(x - w/2, tr, w, label="Treatment (gun present)",
           color="#c1432c", yerr=err(tr, tr_ci), capsize=4)
    ax.bar(x + w/2, cr, w, label="Control (gun absent)",
           color="#4a72b0", yerr=err(cr, cr_ci), capsize=4)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Distractor intrusion rate")
    ax.set_title("Chekhov's Gun: lexical intrusion by task regime (Qwen3-8B)")
    ax.legend(frameon=False, fontsize=9); ax.set_ylim(0, 1)
    for i, v in enumerate(tr):
        ax.text(x[i]-w/2, v+0.02, f"{v:.2f}", ha="center", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_main_rates.png"), dpi=160)
    plt.close(fig)
    print("wrote fig1_main_rates.png")

# ---- Figure 2: distance decay (behavioral, generative) ----
def rate_by_distance(rows):
    d = {}
    for r in rows:
        dd = r["distance"]
        d.setdefault(dd, [0, 0, 0])  # [treat_hits, ctrl_hits, n]
        d[dd][0] += int(r["treatment"]["intrusion"])
        d[dd][1] += int(r["control"]["intrusion"])
        d[dd][2] += 1
    return d

if "full_gen" in beh:
    d = rate_by_distance(beh["full_gen"]["rows"])
    xs = sorted(d)
    tr = [d[k][0]/d[k][2] for k in xs]; cr = [d[k][1]/d[k][2] for k in xs]
    tci = [wilson_ci(d[k][0], d[k][2]) for k in xs]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    import numpy as np
    ax.plot(xs, tr, "-o", color="#c1432c", label="Treatment")
    ax.fill_between(xs, [c[0] for c in tci], [c[1] for c in tci],
                    color="#c1432c", alpha=0.15)
    ax.plot(xs, cr, "--s", color="#4a72b0", label="Control")
    ax.set_xlabel("Turn distance between planting and query")
    ax.set_ylabel("Intrusion rate")
    ax.set_title("Intrusion decays with distance (generative dialogue)")
    ax.legend(frameon=False); ax.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_distance_decay.png"), dpi=160)
    plt.close(fig); print("wrote fig2_distance_decay.png")

# ---- Figure 3: mechanistic probability elevation ----
mech = {}
for key, label in COND:
    s = load_summary(os.path.join(BASE, "mech", key, "summary.json"))
    if s:
        mech[key] = s
if mech:
    import numpy as np
    order = [k for k, _ in COND if k in mech]
    labels = [beh[k]["label"] if k in beh else k for k in order]
    delta = [mech[k]["mean_delta_logp"] for k in order]
    ci = [mech[k].get("delta_logp_ci95", [d, d]) for k, d in zip(order, delta)]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    lo = [d - c[0] for d, c in zip(delta, ci)]
    hi = [c[1] - d for d, c in zip(delta, ci)]
    ax.bar(x, delta, 0.55, color="#7a3b8f", yerr=[lo, hi], capsize=5)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Δ log P(distractor)  treatment − control  (nats)")
    ax.set_title("Mechanistic priming persists in ALL regimes (the loaded gun)")
    for i, v in enumerate(delta):
        ax.text(x[i], v + 0.15, f"{v:.1f}", ha="center", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_mech_prob.png"), dpi=160)
    plt.close(fig); print("wrote fig3_mech_prob.png")

    # Figure 4: mechanistic delta_logp by distance
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for key in order:
        dc = mech[key].get("delta_by_distance", {})
        if dc:
            xs = sorted(int(k) for k in dc)
            ys = [dc[str(k)] for k in xs]
            ax.plot(xs, ys, "-o", label=beh[key]["label"].replace("\n", " ")
                    if key in beh else key)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("Turn / sentence distance")
    ax.set_ylabel("Δ log P(distractor)  (nats)")
    ax.set_title("Priming strength decays with distance")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_mech_distance.png"), dpi=160)
    plt.close(fig); print("wrote fig4_mech_distance.png")

# ---- Figure 6: decoding-intervention temperature sweep ----
import numpy as np
dec = {}
for key in ("gen", "scoped"):
    p = os.path.join(BASE, "decoding", key, "summary.json")
    if os.path.exists(p):
        dec[key] = json.load(open(p))
if dec:
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    colors = {"gen": "#c1432c", "scoped": "#7a3b8f"}
    labels = {"gen": "generative dialogue", "scoped": "scoped dialogue"}
    for key, s in dec.items():
        xs = [c["temperature"] for c in s["curve"]]
        tr = [c["treatment_intrusion_rate"] for c in s["curve"]]
        cr = [c["control_intrusion_rate"] for c in s["curve"]]
        ax.plot(xs, tr, "-o", color=colors.get(key, None), label=f"{labels.get(key,key)} (treat)")
        ax.plot(xs, cr, "--s", color=colors.get(key, None), alpha=0.5,
                label=f"{labels.get(key,key)} (ctrl)")
    ax.set_xlabel("Sampling temperature")
    ax.set_ylabel("Intrusion rate")
    ax.set_title("Sampling temperature does not modulate intrusion")
    ax.legend(frameon=False, fontsize=8); ax.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig6_decoding_sweep.png"), dpi=160)
    plt.close(fig); print("wrote fig6_decoding_sweep.png")

# ---- Semantic intrusion (lower-bound tightening) ----
sem = {}
for key, _ in COND:
    short = key.replace("full_", "")
    p = os.path.join(BASE, "semantic", short, "summary.json")
    if os.path.exists(p):
        sem[key] = json.load(open(p))

# ---- Consolidated table ----
table = {"behavioral": {k: beh[k]["summary"] for k in beh},
         "mechanistic": mech, "decoding": dec, "semantic": sem}
# ablations if present
for ab in glob.glob(os.path.join(BASE, "ablation", "*", "summary.json")):
    name = os.path.basename(os.path.dirname(ab))
    table.setdefault("ablation", {})[name] = json.load(open(ab))
with open(os.path.join(BASE, "..", "results_table.json"), "w") as f:
    json.dump(table, f, indent=2)
print("wrote results_table.json")
