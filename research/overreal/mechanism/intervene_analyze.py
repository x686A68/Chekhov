"""Summarize the presence judgments: presence rate per condition and family,
paired difference to the matching baseline (S conditions vs baseline, P
conditions vs P_baseline), bootstrap over items.
Usage: python intervene_analyze.py --model sd35l
"""
import argparse, collections, glob, json, os, random
ROOT = os.path.dirname(os.path.abspath(__file__))
FAMS = ["cancellation", "attribution", "figurative", "perspectival"]
ORDER = ["baseline", "cue_x2", "cue_x4", "cue_x8", "cue_x8_all", "tgt_d2", "tgt_d4", "tgt_d8", "rand_x8", "func_x8", "P_baseline", "P_rep_x8", "P_tgt_d8"]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--json", default="")
    a = ap.parse_args(); random.seed(0)
    rows = [json.loads(l) for f in glob.glob(os.path.join(ROOT, "intervene", a.model, "judge_*.jsonl")) for l in open(f)]
    pres = {(r["cond"], r["item_id"], r["seed"]): r["present"] for r in rows}
    fam = {r["item_id"]: r["family"] for r in rows}
    conds = [c for c in ORDER if any(k[0] == c for k in pres)]
    def cell(cond, f):
        keys = [k for k in pres if k[0] == cond and (f is None or fam[k[1]] == f)]
        if not keys: return None
        base = "P_baseline" if cond.startswith("P_") and cond != "P_baseline" else "baseline"
        paired = [(pres[k], pres.get((base, k[1], k[2]))) for k in keys]
        paired = [(x, y) for x, y in paired if y is not None]
        rate = sum(pres[k] for k in keys) / len(keys)
        if not paired or cond in ("baseline", "P_baseline"): return dict(n=len(keys), rate=rate)
        diffs_by_item = collections.defaultdict(list)
        for k, (x, y) in zip([k for k in keys if (base, k[1], k[2]) in pres], paired): diffs_by_item[k[1]].append(int(x) - int(y))
        items = list(diffs_by_item); d = sum(sum(v) for v in diffs_by_item.values()) / sum(len(v) for v in diffs_by_item.values())
        bs = []
        for _ in range(2000):
            smp = [diffs_by_item[items[random.randrange(len(items))]] for _ in items]
            flat = [x for v in smp for x in v]; bs.append(sum(flat) / len(flat))
        bs.sort(); return dict(n=len(keys), rate=rate, diff=d, lo=bs[int(.025 * len(bs))], hi=bs[int(.975 * len(bs))])
    out = {}
    print(f"{'cond':12s}" + "".join(f"{f[:13]:>30s}" for f in FAMS + ["all"]))
    for c in conds:
        line = f"{c:12s}"
        for f in FAMS + [None]:
            r = cell(c, f); out[(c, f or "all")] = r
            if r is None: line += f"{'---':>30s}"; continue
            s = f"{r['rate']:.2f}" + (f" {r['diff']:+.2f}[{r['lo']:+.2f},{r['hi']:+.2f}]" if "diff" in r else "") + f" n={r['n']}"
            line += f"{s:>30s}"
        print(line)
    if a.json: json.dump({f"{c}|{f}": v for (c, f), v in out.items()}, open(a.json, "w"), indent=1)

if __name__ == "__main__":
    main()
