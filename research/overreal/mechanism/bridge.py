"""Bridge from the encoder probe to the generation results: does the probe's
held-out margin on an S item predict how often the generator that uses that
encoder over-realizes the item?

Margins come from results/text_encoder.json (span feature, deployed layer).
Over-realization labels come from the Opus automatic annotation of the
raw-prompt generations (data/overreal_v1/auto/final__claude-opus-5-api__sample.jsonl):
an image counts as over-realized when its label is disruptive or silent,
correct when integrated or withheld, skipped when other. The item rate is the
mean over its annotated seeds.

Encoder -> generator pairs: clip_l/clip_g/t5 -> sd35l; clip_l (pooled) and
t5 -> flux; qwen -> qwen-image.

Prints Spearman rho (margin vs item rate) per family and pooled, and the AUC
of the margin for separating items that always over-realize from items that
never do.

Usage: python bridge.py
"""
import json
import os
import re
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(ROOT, "..", "..", ".."))
RES_DIR = os.path.join(ROOT, "results")
AUTO = os.path.join(REPO, "data", "overreal_v1", "auto", "final__claude-opus-5-api__sample.jsonl")

PAIRS = [("clip_l", "span", "sd35l"), ("clip_g", "span", "sd35l"), ("t5", "span", "sd35l"),
         ("clip_l", "pooled", "flux"), ("t5", "span", "flux"), ("qwen", "span", "qwen-image")]
FAMILIES = ["cancellation", "attribution", "figurative", "perspectival"]


def item_rates():
    per = defaultdict(list)
    for line in open(AUTO):
        r = json.loads(line)
        m = re.match(r"(.+)/gen_(.+)_raw_s\d+$", r["image_id"])
        if not m:
            continue
        item, model = m.group(1), m.group(2)
        lab = r.get("label")
        if lab in ("disruptive", "silent"):
            per[(model, item)].append(1.0)
        elif lab in ("integrated", "withheld"):
            per[(model, item)].append(0.0)
    return {k: float(np.mean(v)) for k, v in per.items()}


def main():
    res = {}
    for enc in ["clip_l", "clip_g", "t5", "qwen"]:
        p = os.path.join(RES_DIR, f"text_encoder_{enc}.json")
        if os.path.exists(p):
            res.update(json.load(open(p)))
    rates = item_rates()
    print(f"{'encoder':<8}{'feat':<8}{'model':<12}{'family':<14}{'n':>5}{'rho':>8}{'p':>8}{'AUC':>7}")
    for enc, feat, model in PAIRS:
        if enc not in res or feat not in res[enc]:
            continue
        ids = res[enc]["item_ids"]
        margins = {ids[int(i)]: m for i, m in res[enc][feat]["margin_S"].items()}
        rows = [(iid.split("/")[0], margins[iid], rates[(model, iid)])
                for iid in ids if iid in margins and (model, iid) in rates]
        for fam in FAMILIES + ["pooled"]:
            sel = [(m, r) for f, m, r in rows if fam == "pooled" or f == fam]
            if len(sel) < 10:
                continue
            m = np.array([s[0] for s in sel])
            r = np.array([s[1] for s in sel])
            rho, p = spearmanr(m, r)
            ext = (r == 1) | (r == 0)
            auc = roc_auc_score((r[ext] == 1).astype(int), -m[ext]) if len(set(r[ext])) == 2 else float("nan")
            print(f"{enc:<8}{feat:<8}{model:<12}{fam:<14}{len(sel):>5}{rho:>8.2f}{p:>8.3f}{auc:>7.2f}")


if __name__ == "__main__":
    main()
