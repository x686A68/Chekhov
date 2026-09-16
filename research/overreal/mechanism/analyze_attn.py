"""Analyze the attention records of attn_<model>.py.

Per image, the target's attention is the mean over blocks and steps of the
share of image-to-text attention that lands on the target tokens:
  share_target = sum_{j in target} mass[j] / sum_{j real} mass[j]
(the raw mass is reported alongside). The same for the cue tokens (the words
the rewrite replaced) and for the remaining real tokens.

Three readings:
  1. S vs P: does the target get the same attention whether or not the
     prompt calls for it? Paired log-ratio log(share_S / share_P) per item
     and seed, per family, with a bootstrap 95% interval.
  2. Within S: does the target's attention separate images where the target
     appeared (disruptive or silent, by the Opus annotation) from withheld
     ones? AUC per family; also for the peak of the target map.
  3. The cue tokens: their per-token share against the other real tokens,
     and their share in withheld versus over-realized images (AUC).

Usage: python analyze_attn.py --model flux
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(ROOT, "..", "..", ".."))
AUTO = os.path.join(REPO, "data", "overreal_v1", "auto", "final__claude-opus-5-api__sample.jsonl")
FAMILIES = ["cancellation", "attribution", "figurative", "perspectival"]
MODEL_KEY = {"flux": "flux", "sd35l": "sd35l", "qwen-image": "qwen-image"}


def labels(model):
    lab = {}
    for line in open(AUTO):
        r = json.loads(line)
        m = re.match(r"(.+)/gen_(.+)_raw_s(\d+)$", r["image_id"])
        if m and m.group(2) == MODEL_KEY[model]:
            lab[(m.group(1), int(m.group(3)))] = r.get("label")
    return lab


def summarize(z):
    mass = z["mass"].astype(np.float32)           # [blocks, steps, n_txt]
    n = int(z["n_real"])
    t = z["target_idx"]
    c = z["cue_idx"]
    per_tok = mass.mean((0, 1))                    # mean over blocks and steps
    real = per_tok[:n]
    total = real.sum()
    other = np.ones(n, bool)
    other[t] = False
    if len(c):
        other[c] = False
    out = {
        "target_mass": float(per_tok[t].sum()),
        "target_share": float(per_tok[t].sum() / total),
        "target_share_early": float(mass[:, : max(1, mass.shape[1] // 4)].mean((0, 1))[t].sum()
                                    / mass[:, : max(1, mass.shape[1] // 4)].mean((0, 1))[:n].sum()),
        "cue_share_tok": float(per_tok[c].mean() / total) if len(c) else np.nan,
        "other_share_tok": float(real[other].mean() / total) if other.any() else np.nan,
        "target_share_tok": float(per_tok[t].mean() / total),
        "map_peak": float(z["map"].astype(np.float32)[:2].mean(0).max()),
        "text_total": float(total),
        "profile": mass[:, :, t].sum(-1) / mass[:, :, :n].sum(-1),  # [blocks, steps] share
    }
    return out


def boot_ci(x, n=2000, seed=0):
    rng = np.random.RandomState(seed)
    x = np.asarray(x)
    if len(x) == 0:
        return (np.nan, np.nan)
    b = [rng.choice(x, len(x), replace=True).mean() for _ in range(n)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="flux")
    args = ap.parse_args()
    d = os.path.join(ROOT, "attn", args.model)
    recs = {}
    for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
        z = np.load(f)
        key = (str(z["item_id"]), str(z["side"]), int(z["seed"]))
        recs[key] = summarize(z) | {"family": str(z["family"])}
    print(f"{len(recs)} records")
    lab = labels(args.model)

    # ---- reading 1: S vs P -------------------------------------------------
    print("\n[1] target attention, S vs P (paired by item and seed)")
    print(f"{'family':<14}{'n':>5}{'share S':>9}{'share P':>9}{'log-ratio':>11}{'95% CI':>17}{'S<P':>7}")
    rows1 = []
    for fam in FAMILIES + ["all"]:
        lr, ss, pp = [], [], []
        for (item, side, seed), r in recs.items():
            if side != "S" or (fam != "all" and r["family"] != fam):
                continue
            q = recs.get((item, "P", seed))
            if q is None:
                continue
            lr.append(np.log(r["target_share"] / q["target_share"]))
            ss.append(r["target_share"]); pp.append(q["target_share"])
        if not lr:
            continue
        lo, hi = boot_ci(lr)
        print(f"{fam:<14}{len(lr):>5}{np.mean(ss):>9.4f}{np.mean(pp):>9.4f}{np.mean(lr):>11.3f}"
              f"{f'[{lo:.3f}, {hi:.3f}]':>17}{np.mean(np.array(lr) < 0):>7.2f}")
        rows1.append((fam, len(lr), np.mean(ss), np.mean(pp), np.mean(lr), lo, hi))

    # ---- reading 2: within S, attention vs outcome ------------------------
    print("\n[2] within S: does the target's attention predict its appearance? (DO+SO vs W)")
    print(f"{'family':<14}{'n_over':>7}{'n_with':>7}{'AUC share':>10}{'AUC early':>10}{'AUC peak':>9}{'AUC mass':>9}")
    for fam in FAMILIES + ["all"]:
        y, s, e, p, m = [], [], [], [], []
        for (item, side, seed), r in recs.items():
            if side != "S" or (fam != "all" and r["family"] != fam):
                continue
            l = lab.get((item, seed))
            if l in ("disruptive", "silent"):
                y.append(1)
            elif l == "withheld":
                y.append(0)
            else:
                continue
            s.append(r["target_share"]); e.append(r["target_share_early"])
            p.append(r["map_peak"]); m.append(r["target_mass"])
        if len(set(y)) < 2:
            print(f"{fam:<14}{sum(y):>7}{len(y)-sum(y):>7}   (too few)")
            continue
        print(f"{fam:<14}{sum(y):>7}{len(y)-sum(y):>7}{roc_auc_score(y, s):>10.2f}{roc_auc_score(y, e):>10.2f}"
              f"{roc_auc_score(y, p):>9.2f}{roc_auc_score(y, m):>9.2f}")

    # ---- reading 3: the cue tokens ----------------------------------------
    print("\n[3] cue tokens (the words the rewrite replaced), per-token share of text attention")
    print(f"{'family':<14}{'n':>5}{'cue':>9}{'other':>9}{'target':>9}{'cue/other':>10}{'AUC cue W>over':>16}")
    for fam in FAMILIES + ["all"]:
        cue, oth, tgt, y, cy = [], [], [], [], []
        for (item, side, seed), r in recs.items():
            if side != "S" or (fam != "all" and r["family"] != fam) or np.isnan(r["cue_share_tok"]):
                continue
            cue.append(r["cue_share_tok"]); oth.append(r["other_share_tok"]); tgt.append(r["target_share_tok"])
            l = lab.get((item, seed))
            if l in ("disruptive", "silent"):
                y.append(0); cy.append(r["cue_share_tok"])
            elif l == "withheld":
                y.append(1); cy.append(r["cue_share_tok"])
        if not cue:
            continue
        auc = roc_auc_score(y, cy) if len(set(y)) == 2 else np.nan
        print(f"{fam:<14}{len(cue):>5}{np.mean(cue):>9.4f}{np.mean(oth):>9.4f}{np.mean(tgt):>9.4f}"
              f"{np.mean(cue)/np.mean(oth):>10.2f}{auc:>16.2f}")

    # ---- profile for the appendix -----------------------------------------
    prof = defaultdict(list)
    for (item, side, seed), r in recs.items():
        prof[side].append(r["profile"])
    out = {s: np.mean(np.stack(v), 0).tolist() for s, v in prof.items() if v}
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", f"attn_profile_{args.model}.json"), "w") as fp:
        json.dump(out, fp)
    with open(os.path.join(ROOT, "results", f"attn_summary_{args.model}.json"), "w") as fp:
        json.dump({"|".join(map(str, k)): {kk: vv for kk, vv in v.items() if kk != "profile"}
                   for k, v in recs.items()}, fp)


if __name__ == "__main__":
    main()
