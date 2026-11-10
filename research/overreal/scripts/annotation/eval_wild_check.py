"""Judge-vs-human agreement on the in-the-wild human-check sample (3P6s, Q2).

Reads research/overreal/wild/human_check_sample.jsonl (judge labels) and the
local shards data/wild_check_annotations/*.jsonl (human labels; latest record
per image and annotator wins; a two-label record counts as agreeing when it
contains the judge label, as in eval_auto.py). Reports, per family and
overall: n, exact agreement, Cohen's kappa on the four labels (other kept),
kappa on over-realized vs. correct, and the judge-vs-human confusion matrix.
Writes research/overreal/wild/human_check_results.md.
"""
import collections
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SAMPLE = ROOT / "research" / "overreal" / "wild" / "human_check_sample.jsonl"
ANN_DIR = ROOT / "data" / "wild_check_annotations"
OUT = ROOT / "research" / "overreal" / "wild" / "human_check_results.md"
LABS = ["disruptive", "silent", "integrated", "withheld", "other"]
OR = {"disruptive", "silent"}
FAM_NAME = {"cancellation": "Existence-canceling", "attribution": "Mental-state",
            "figurative": "Figurative", "perspectival": "Perspectival"}


def kappa(pairs):
    """Cohen's kappa for (a, b) label pairs."""
    n = len(pairs)
    if n == 0:
        return float("nan")
    po = sum(a == b for a, b in pairs) / n
    ca, cb = collections.Counter(a for a, _ in pairs), collections.Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def main():
    sample = [json.loads(l) for l in open(SAMPLE, encoding="utf-8")]
    latest = {}
    for f in sorted(glob.glob(str(ANN_DIR / "*.jsonl"))):
        for l in open(f, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                latest[(r["image_id"], r["annotator"])] = r
    human = collections.defaultdict(dict)
    for (iid, ann), r in sorted(latest.items(), key=lambda kv: kv[1].get("ts", "")):
        human[iid][ann] = r["labels"]

    rows = []
    for r in sample:
        for ann, labels in human.get(r["image_id"], {}).items():
            j = r["judge_label"]
            h = j if j in labels else labels[0]      # two-label record: credit the judge if it matches
            rows.append((r["family"], ann, j, h, r["image_id"]))
    lines = ["# In-the-wild human check: judge vs. human", "",
             f"Sample: {len(sample)} images; annotated so far: {len({x[4] for x in rows})} "
             f"(annotators: {sorted({x[1] for x in rows})})", "",
             "| Family | n | exact agreement | κ, 4 labels | κ, over-realized vs. correct |",
             "|---|---|---|---|---|"]
    for fam in ["cancellation", "attribution", "figurative", "perspectival", "all"]:
        sub = [x for x in rows if fam == "all" or x[0] == fam]
        if not sub:
            continue
        pairs = [(j, h) for _, _, j, h, _ in sub]
        agree = sum(j == h for j, h in pairs) / len(pairs)
        k4 = kappa(pairs)
        k2 = kappa([(j in OR, h in OR) for j, h in pairs])
        lines.append(f"| {FAM_NAME.get(fam, 'All')} | {len(sub)} | {agree:.2f} | {k4:.2f} | {k2:.2f} |")
    lines += ["", "Confusion matrix (rows: judge, columns: human), all families:", "",
              "| judge \\ human | " + " | ".join(LABS) + " |", "|---|" + "---|" * len(LABS)]
    conf = collections.Counter((j, h) for _, _, j, h, _ in rows)
    for j in LABS:
        lines.append(f"| {j} | " + " | ".join(str(conf[(j, h)]) for h in LABS) + " |")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
