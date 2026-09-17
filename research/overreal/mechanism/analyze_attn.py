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


_SD35_TOK = {}


def real_indices(z, model):
    if "real_idx" in z.files:
        return z["real_idx"]
    if model != "sd35l":
        return np.arange(int(z["n_real"]))
    # older sd35l records: recompute the real positions from the tokenizers
    if not _SD35_TOK:
        from transformers import CLIPTokenizer, T5TokenizerFast
        _SD35_TOK["clip"] = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-3.5-large", subfolder="tokenizer")
        _SD35_TOK["t5"] = T5TokenizerFast.from_pretrained("stabilityai/stable-diffusion-3.5-large", subfolder="tokenizer_3")
    text = str(z["prompt"])
    e1 = _SD35_TOK["clip"](text, padding="max_length", max_length=77, truncation=True, return_offsets_mapping=True)
    e2 = _SD35_TOK["t5"](text, padding="max_length", max_length=256, truncation=True, return_offsets_mapping=True)
    return np.array([i for i, (a, b) in enumerate(e1["offset_mapping"]) if b > a]
                    + [77 + i for i, (a, b) in enumerate(e2["offset_mapping"]) if b > a])


def summarize(z, model="flux"):
    mass = z["mass"].astype(np.float32)           # [blocks, steps, n_txt]
    t = z["target_idx"]
    c = z["cue_idx"]
    real_idx = real_indices(z, model)
    per_tok = mass.mean((0, 1))                    # mean over blocks and steps
    total = per_tok[real_idx].sum()
    other = np.ones(len(per_tok), bool)
    other[t] = False
    if len(c):
        other[c] = False
    other_idx = np.array([i for i in real_idx if other[i]])
    early = mass[:, : max(1, mass.shape[1] // 4)].mean((0, 1))
    out = {
        "target_mass": float(per_tok[t].sum()),
        "target_share": float(per_tok[t].sum() / total),
        "target_share_early": float(early[t].sum() / early[real_idx].sum()),
        "cue_share_tok": float(per_tok[c].mean() / total) if len(c) else np.nan,
        "other_share_tok": float(per_tok[other_idx].mean() / total) if len(other_idx) else np.nan,
        "target_share_tok": float(per_tok[t].mean() / total),
        "map_peak": float(z["map"].astype(np.float32)[:2].mean(0).max()),
        "text_total": float(total),
        "profile": _bin_steps(mass[:, :, t].sum(-1) / mass[:, :, real_idx].sum(-1), 10),  # [blocks, 10]
    }
    return out


def _bin_steps(x, nb):
    """Average the step axis of [blocks, steps] into nb equal bins."""
    T = x.shape[1]
    edges = np.linspace(0, T, nb + 1).astype(int)
    return np.stack([x[:, a:max(b, a + 1)].mean(1) for a, b in zip(edges[:-1], edges[1:])], 1)


def boot_ci(x, n=2000, seed=0):
    rng = np.random.RandomState(seed)
    x = np.asarray(x)
    if len(x) == 0:
        return (np.nan, np.nan)
    b = [rng.choice(x, len(x), replace=True).mean() for _ in range(n)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def expanded(args, recs, lab):
    """Expanded prompts (side X) against the raw prompt (side S) of the same
    item and seed: the target's per-token share relative to the other real
    tokens (length-free), and whether it still predicts appearance."""
    d = os.path.join(ROOT, "attn", f"{args.model}_{args.cond}")
    xrecs = {}
    for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
        z = np.load(f)
        xrecs[(str(z["item_id"]), int(z["seed"]))] = summarize(z, args.model) | {"family": str(z["family"])}
    xlab = {}
    for line in open(AUTO):
        r = json.loads(line)
        m = re.match(rf"(.+)/gen_{MODEL_KEY[args.model]}_{args.cond}_s(\d+)$", r["image_id"])
        if m:
            xlab[(m.group(1), int(m.group(2)))] = r.get("label")
    print(f"\n[expanded: {args.cond}] {len(xrecs)} records; target per-token share relative to other tokens")
    print(f"{'family':<14}{'n':>5}{'raw':>8}{'expanded':>10}{'log-ratio':>11}{'95% CI':>17}"
          f"{'n_over':>7}{'n_with':>7}{'AUC':>6}")
    for fam in FAMILIES + ["all"]:
        lr, a, b, y, sc = [], [], [], [], []
        for (item, seed), x in xrecs.items():
            if fam != "all" and x["family"] != fam:
                continue
            rel_x = x["target_share_tok"] / x["other_share_tok"]
            sraw = recs.get((item, "S", seed))
            if sraw is not None:
                rel_s = sraw["target_share_tok"] / sraw["other_share_tok"]
                lr.append(np.log(rel_x / rel_s)); a.append(rel_s); b.append(rel_x)
            l = xlab.get((item, seed))
            if l in ("disruptive", "silent"):
                y.append(1); sc.append(rel_x)
            elif l == "withheld":
                y.append(0); sc.append(rel_x)
        if not lr and not y:
            continue
        lo, hi = boot_ci(lr) if lr else (np.nan, np.nan)
        auc = roc_auc_score(y, sc) if len(set(y)) == 2 else np.nan
        print(f"{fam:<14}{len(lr):>5}{np.mean(a) if a else np.nan:>8.2f}{np.mean(b) if b else np.nan:>10.2f}"
              f"{np.mean(lr) if lr else np.nan:>11.3f}{f'[{lo:.3f}, {hi:.3f}]':>17}"
              f"{sum(y):>7}{len(y)-sum(y):>7}{auc:>6.2f}")


def table(args, recs, lab):
    """Per-family and family-mean rows of the paper tables: per-token shares
    of target / cue / other words under S and P, the S/P target ratio, and the
    AUCs (early map peak for the target; cue share for withheld)."""
    rows = {}
    for fam in FAMILIES:
        S = [(k, r) for k, r in recs.items() if k[1] == "S" and r["family"] == fam]
        P = {k[0:1] + k[2:]: r for k, r in recs.items() if k[1] == "P" and r["family"] == fam}
        st, sc, so, pt, pc, po, lr, lrc = [], [], [], [], [], [], [], []
        y, pk, cy = [], [], []
        for (item, side, seed), r in S:
            q = P.get((item, seed))
            if q is None:
                continue
            st.append(r["target_share_tok"]); sc.append(r["cue_share_tok"]); so.append(r["other_share_tok"])
            pt.append(q["target_share_tok"]); pc.append(q["cue_share_tok"]); po.append(q["other_share_tok"])
            lr.append(np.log(r["target_share"] / q["target_share"]))
            if not (np.isnan(r["cue_share_tok"]) or np.isnan(q["cue_share_tok"])) and q["cue_share_tok"] > 0:
                lrc.append(np.log(r["cue_share_tok"] / q["cue_share_tok"]))
            l = lab.get((item, seed))
            if l in ("disruptive", "silent"):
                y.append(1); pk.append(r["map_peak"]); cy.append(r["cue_share_tok"])
            elif l == "withheld":
                y.append(0); pk.append(r["map_peak"]); cy.append(r["cue_share_tok"])
        auc = roc_auc_score(y, pk) if len(set(y)) == 2 else np.nan
        ok = [i for i, v in enumerate(cy) if not np.isnan(v)]
        auc_c = roc_auc_score([1 - y[i] for i in ok], [cy[i] for i in ok]) if len(set(y[i] for i in ok)) == 2 else np.nan
        rows[fam] = dict(n=len(lr), s_t=np.nanmean(st), s_c=np.nanmean(sc), s_o=np.nanmean(so),
                         p_t=np.nanmean(pt), p_c=np.nanmean(pc), p_o=np.nanmean(po),
                         ratio=float(np.exp(np.mean(lr))), ratio_c=float(np.exp(np.mean(lrc))),
                         n_c=len(lrc), auc=auc, auc_c=auc_c)
    rows["mean"] = {k: (sum(rows[f]["n"] for f in FAMILIES) if k == "n" else float(np.mean([rows[f][k] for f in FAMILIES])))
                    for k in rows[FAMILIES[0]]}
    print(f"\n[table] {args.model}: per-token share (% of image-to-text attention on real tokens)")
    print(f"{'family':<14}{'n':>5}{'S tgt':>7}{'S cue':>7}{'S oth':>7}{'P tgt':>7}{'P cue':>7}{'P oth':>7}{'rho_t':>7}{'rho_c':>7}{'n_c':>5}{'AUC':>6}{'AUCcue':>7}")
    for fam in FAMILIES + ["mean"]:
        r = rows[fam]
        print(f"{fam:<14}{r['n']:>5}{100*r['s_t']:>7.2f}{100*r['s_c']:>7.2f}{100*r['s_o']:>7.2f}"
              f"{100*r['p_t']:>7.2f}{100*r['p_c']:>7.2f}{100*r['p_o']:>7.2f}{r['ratio']:>7.2f}{r['ratio_c']:>7.2f}{int(r['n_c']):>5}{r['auc']:>6.2f}{r['auc_c']:>7.2f}")
    with open(os.path.join(ROOT, "results", f"attn_table_{args.model}.json"), "w") as fp:
        json.dump(rows, fp, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="flux")
    ap.add_argument("--cond", default="", help="expanded-prompt run to compare with raw: qwen or ideogram")
    ap.add_argument("--table", action="store_true", help="print the paper-table rows and exit")
    args = ap.parse_args()
    d = os.path.join(ROOT, "attn", args.model)
    recs = {}
    for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
        z = np.load(f)
        key = (str(z["item_id"]), str(z["side"]), int(z["seed"]))
        recs[key] = summarize(z, args.model) | {"family": str(z["family"])}
    print(f"{len(recs)} records")
    lab = labels(args.model)
    if args.cond:
        return expanded(args, recs, lab)
    if args.table:
        return table(args, recs, lab)

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
