"""Linear probes on the text-encoder features: does the encoder output tell a
space-builder prompt (S, label 1) from its plain-mention control (P, label 0)?

One probe per encoder and feature, trained on all four families together with
family-balanced sample weights, tested with GroupKFold (5 folds) grouped by
the target word so no target appears in both train and test. Accuracy is
reported per family (held-out predictions) and pooled, with a bootstrap 95%
interval, a label-shuffle null, and a training-free mass-mean probe.

Features (see extract_features.py): span (target tokens), ctrl (control
word), mean (prompt mean), pooled / proj (CLIP pooled vectors).
Deployed layer: hidden_states[-2] for CLIP (what SD3 feeds the backbone),
last layer for T5 and Qwen. The layer sweep of the span probe goes to
results/layer_sweep.json.

Usage: python probe.py [--encoders clip_l clip_g t5 qwen] [--no-sweep]
"""
import argparse
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.abspath(__file__))
FEAT = os.path.join(ROOT, "features")
RES = os.path.join(ROOT, "results")
PAIRS = os.path.join(ROOT, "pairs.jsonl")

FAMILIES = ["cancellation", "attribution", "figurative", "perspectival"]
DEPLOYED = {"clip_l": -2, "clip_g": -2, "t5": -1, "qwen": -1}
C_GRID = [0.01, 0.1, 1.0, 10.0]
N_FOLDS = 5
N_BOOT = 1000
N_NULL = 100
SEED = 0


def group_key(target):
    t = target.lower().strip().strip('"').strip("'").rstrip(".").strip()
    t = re.sub(r"[^a-z ]", " ", t)
    words = [w for w in t.split() if len(w) > 2]
    return " ".join(words) if words else t


def load(encoder):
    z = np.load(os.path.join(FEAT, f"{encoder}.npz"))
    meta = {r["item_id"]: r for r in (json.loads(l) for l in open(PAIRS)) if not r["exclude"]}
    ids = list(z["item_ids"])
    fam = np.array([meta[i]["family"] for i in ids])
    groups = np.array([group_key(meta[i]["target"]) for i in ids])
    return z, ids, fam, groups


def family_weights(fam):
    c = Counter(fam)
    return np.array([1.0 / c[f] for f in fam]) * len(fam) / len(c)


def make_xy(z, key, layer, fam, groups):
    a, b = z[f"{key}_S"], z[f"{key}_P"]
    if a.ndim == 3:
        a, b = a[:, layer], b[:, layer]
    a, b = a.astype(np.float32), b.astype(np.float32)
    keep = ~(np.isnan(a).any(1) | np.isnan(b).any(1))
    X = np.concatenate([a[keep], b[keep]])
    y = np.concatenate([np.ones(keep.sum()), np.zeros(keep.sum())]).astype(int)
    f = np.concatenate([fam[keep], fam[keep]])
    g = np.concatenate([groups[keep], groups[keep]])
    idx = np.concatenate([np.where(keep)[0], np.where(keep)[0]])  # item index
    return X, y, f, g, idx


def lr(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))


def cv_predict(X, y, f, g, C=None, seed=SEED, inner=True):
    """Held-out decision values and predictions. C chosen by inner GroupKFold
    on each training fold when C is None."""
    rng = np.random.RandomState(seed)
    # shuffle group order so folds are not tied to family order
    ug = np.unique(g)
    perm = {gg: i for i, gg in enumerate(rng.permutation(ug))}
    gnum = np.array([perm[gg] for gg in g])
    w = family_weights(f)
    dec = np.zeros(len(y))
    mm = np.zeros(len(y))  # mass-mean probe prediction
    chosen = []
    for tr, te in GroupKFold(N_FOLDS).split(X, y, gnum):
        Cbest = C
        if Cbest is None:
            best = None
            for c in C_GRID:
                accs = []
                for itr, ite in GroupKFold(3).split(X[tr], y[tr], gnum[tr]):
                    m = lr(c).fit(X[tr][itr], y[tr][itr], logisticregression__sample_weight=w[tr][itr])
                    accs.append(((m.predict(X[tr][ite]) == y[tr][ite]) * w[tr][ite]).sum() / w[tr][ite].sum())
                if best is None or np.mean(accs) > best[0]:
                    best = (np.mean(accs), c)
            Cbest = best[1]
        chosen.append(Cbest)
        m = lr(Cbest).fit(X[tr], y[tr], logisticregression__sample_weight=w[tr])
        dec[te] = m.decision_function(X[te])
        # mass-mean: direction = mean(S) - mean(P) on standardized train data
        sc = m.named_steps["standardscaler"]
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        d = Xtr[y[tr] == 1].mean(0) - Xtr[y[tr] == 0].mean(0)
        p_tr = Xtr @ d
        thr = 0.5 * (p_tr[y[tr] == 1].mean() + p_tr[y[tr] == 0].mean())
        mm[te] = Xte @ d - thr
    return dec, mm, chosen


def per_family_acc(pred, y, f, idx):
    out = {}
    for fam in FAMILIES:
        sel = f == fam
        if sel.sum() == 0:
            continue
        out[fam] = float((pred[sel] == y[sel]).mean())
    out["pooled"] = float(np.mean([out[k] for k in out]))  # family-balanced
    return out


def bootstrap_ci(pred, y, f, idx, n=N_BOOT, seed=SEED):
    """Resample items (both sides of a pair together) within each family."""
    rng = np.random.RandomState(seed)
    accs = defaultdict(list)
    items_by_fam = {fam: np.unique(idx[f == fam]) for fam in FAMILIES if (f == fam).any()}
    pos = defaultdict(list)
    for k, (i, fam) in enumerate(zip(idx, f)):
        pos[(fam, i)].append(k)
    for _ in range(n):
        fam_acc = []
        for fam, items in items_by_fam.items():
            samp = rng.choice(items, len(items), replace=True)
            ks = np.concatenate([pos[(fam, i)] for i in samp])
            a = (pred[ks] == y[ks]).mean()
            accs[fam].append(a)
            fam_acc.append(a)
        accs["pooled"].append(np.mean(fam_acc))
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in accs.items()}


def null_dist(X, y, f, g, idx, C, n=None, seed=SEED):
    n = n or N_NULL
    """Shuffle labels within items (swap S/P of a random half of the items)."""
    rng = np.random.RandomState(seed)
    accs = []
    items = np.unique(idx)
    for _ in range(n):
        flip = set(rng.choice(items, len(items) // 2, replace=False))
        y2 = np.array([1 - yy if i in flip else yy for yy, i in zip(y, idx)])
        dec, _, _ = cv_predict(X, y2, f, g, C=C, seed=rng.randint(1 << 30))
        accs.append(per_family_acc((dec > 0).astype(int), y2, f, idx)["pooled"])
    return {"mean": float(np.mean(accs)), "p95": float(np.percentile(accs, 95)),
            "max": float(np.max(accs))}


def run_probe(z, key, layer, fam, groups, do_null=True, n_null=None):
    X, y, f, g, idx = make_xy(z, key, layer, fam, groups)
    dec, mm, chosen = cv_predict(X, y, f, g)
    pred = (dec > 0).astype(int)
    C = Counter(chosen).most_common(1)[0][0]
    res = {
        "n_items": int(len(y) // 2),
        "C": C,
        "acc": per_family_acc(pred, y, f, idx),
        "ci": bootstrap_ci(pred, y, f, idx),
        "massmean_acc": per_family_acc((mm > 0).astype(int), y, f, idx),
    }
    if do_null:
        res["null"] = null_dist(X, y, f, g, idx, C, n=n_null)
    # held-out margins of the S items, for the bridge to generation results
    res["margin_S"] = {int(i): float(d) for i, d, yy in zip(idx, dec, y) if yy == 1}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", default=["clip_l", "clip_g", "t5", "qwen"])
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--no-null", action="store_true")
    ap.add_argument("--n-null", type=int, default=N_NULL)
    args = ap.parse_args()
    os.makedirs(RES, exist_ok=True)

    results = {}
    sweep = {}
    for enc in args.encoders:
        path = os.path.join(FEAT, f"{enc}.npz")
        if not os.path.exists(path):
            print("missing", path)
            continue
        z, ids, fam, groups = load(enc)
        layer = DEPLOYED[enc]
        results[enc] = {"deployed_layer": int(layer), "item_ids": ids}
        keys = ["span", "ctrl", "mean"] + (["pooled", "proj"] if f"pooled_S" in z else [])
        for key in keys:
            r = run_probe(z, key, layer, fam, groups, do_null=not args.no_null, n_null=args.n_null)
            results[enc][key] = r
            a = r["acc"]
            print(f"{enc:7s} {key:7s} C={r['C']:<5} " +
                  " ".join(f"{k[:5]}={a[k]:.2f}" for k in a) +
                  f"  mm={r['massmean_acc']['pooled']:.2f}" +
                  (f"  null95={r['null']['p95']:.2f}" if "null" in r else ""), flush=True)
        if not args.no_sweep:
            n_layers = z["span_S"].shape[1]
            C = results[enc]["span"]["C"]
            sweep[enc] = []
            for L in range(n_layers):
                X, y, f, g, idx = make_xy(z, "span", L, fam, groups)
                dec, _, _ = cv_predict(X, y, f, g, C=C)
                acc = per_family_acc((dec > 0).astype(int), y, f, idx)
                sweep[enc].append(acc)
            print(f"{enc} layer sweep (pooled acc): " +
                  " ".join(f"{a['pooled']:.2f}" for a in sweep[enc]), flush=True)

    tag = "_".join(args.encoders)
    with open(os.path.join(RES, f"text_encoder_{tag}.json"), "w") as fp:
        json.dump(results, fp, indent=1)
    if sweep:
        with open(os.path.join(RES, f"layer_sweep_{tag}.json"), "w") as fp:
            json.dump(sweep, fp, indent=1)
    print("saved to", RES)


if __name__ == "__main__":
    main()
