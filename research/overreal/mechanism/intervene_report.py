"""Markdown summary of the intervention results for all models.
Usage: python intervene_report.py > results/intervention_summary.md
"""
import collections, glob, json, os, random, sys
ROOT = os.path.dirname(os.path.abspath(__file__))
FAMS = [("cancellation", "EC"), ("attribution", "MS"), ("figurative", "Fig"), ("perspectival", "Per")]
ORDER = ["baseline", "cue_x2", "cue_x4", "cue_x8", "cue_x8_all", "tgt_d2", "tgt_d4", "tgt_d8", "rand_x8", "func_x8", "P_baseline", "P_rep_x8", "P_tgt_d8"]
NAMES = {"sd35l": "SD3.5-Large", "flux": "FLUX.1-dev", "qwen-image": "Qwen-Image"}

def load(model):
    rows = [json.loads(l) for f in glob.glob(os.path.join(ROOT, "intervene", model, "judge_*.jsonl")) for l in open(f)]
    pres = {(r["cond"], r["item_id"], r["seed"]): r["present"] for r in rows}
    fam = {r["item_id"]: r["family"] for r in rows}
    return pres, fam

def stat(pres, fam, cond, f):
    random.seed(0)
    keys = [k for k in pres if k[0] == cond and (f is None or fam[k[1]] == f)]
    if not keys: return None
    rate = sum(pres[k] for k in keys) / len(keys)
    if cond in ("baseline", "P_baseline"): return dict(n=len(keys), rate=rate)
    base = "P_baseline" if cond.startswith("P_") else "baseline"
    by = collections.defaultdict(list)
    for k in keys:
        b = pres.get((base, k[1], k[2]))
        if b is not None: by[k[1]].append(int(pres[k]) - int(b))
    if not by: return dict(n=len(keys), rate=rate)
    items = list(by); d = sum(sum(v) for v in by.values()) / sum(len(v) for v in by.values())
    bs = sorted(sum(x for it in (by[items[random.randrange(len(items))]] for _ in items) for x in it) / sum(len(v) for v in by.values()) for _ in range(2000))
    return dict(n=len(keys), rate=rate, diff=d, lo=bs[50], hi=bs[1949])

def table(model):
    pres, fam = load(model)
    conds = [c for c in ORDER if any(k[0] == c for k in pres)]
    out = [f"### {NAMES.get(model, model)}", "", "| condition | " + " | ".join(n for _, n in FAMS) + " | all |", "|---|" + "---|" * (len(FAMS) + 1)]
    for c in conds:
        cells = []
        for f, _ in FAMS + [(None, "all")]:
            r = stat(pres, fam, c, f)
            if r is None: cells.append("—"); continue
            s = f"{r['rate']:.2f}"
            if "diff" in r:
                star = "*" if (r["lo"] > 0 or r["hi"] < 0) else ""
                s += f" ({r['diff']:+.2f}{star})"
            cells.append(s)
        out.append(f"| {c} | " + " | ".join(cells) + " |")
    n_items = len({k[1] for k in pres}); n_img = len([k for k in pres if k[0] not in ("baseline", "P_baseline")])
    out.append(""); out.append(f"{n_items} items, {n_img} intervention images judged. Cells: target presence rate (paired change vs. the matching baseline; * = bootstrap 95% interval over items excludes 0).")
    return "\n".join(out)

if __name__ == "__main__":
    models = sys.argv[1:] or ["sd35l", "flux", "qwen-image"]
    print("\n\n".join(table(m) for m in models))
