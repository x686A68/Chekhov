"""Aggregate the per-chunk visual annotations into one Family 1 label file.

Inputs : dataset/family1/images/inspect_chunks/{chunk,result}_NN.jsonl
Outputs: dataset/family1/annotations_nb2.jsonl  (one line per image, joined with
         the item metadata) and annotations_nb2.stats.json

Annotation protocol (see PILOT_NB2.md): each image was viewed once at normal scale
by a model annotator; no cropping or magnification. Anything not identifiable at
normal viewing scale is recorded realized=false with confidence="low". `realized`
means the target object appears in the image; because every prompt asked for the
target's absence, realized=true is an over-realization.
"""
import glob
import json
import os
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
CH = os.path.join(BASE, "dataset", "family1", "images", "inspect_chunks")
OUT = os.path.join(BASE, "dataset", "family1", "annotations_nb2.jsonl")
STATS = os.path.join(BASE, "dataset", "family1", "annotations_nb2.stats.json")


def main():
    meta = {}
    for f in sorted(glob.glob(os.path.join(CH, "chunk_*.jsonl"))):
        for l in open(f):
            r = json.loads(l)
            meta[r["key"]] = r

    labels, dupes = {}, []
    for f in sorted(glob.glob(os.path.join(CH, "result_*.jsonl"))):
        for l in open(f):
            l = l.strip()
            if not l:
                continue
            r = json.loads(l)
            if r["key"] in labels:
                dupes.append(r["key"])
            labels[r["key"]] = r

    missing = sorted(set(meta) - set(labels))
    extra = sorted(set(labels) - set(meta))

    rows = []
    for k in sorted(meta):
        if k not in labels:
            continue
        m, a = meta[k], labels[k]
        item_id, s = k.rsplit("_s", 1)
        rows.append(dict(
            key=k, item_id=item_id, sample=int(s), target=m["target"],
            plausibility=m["plausibility"], prompt=m["prompt"],
            realized=bool(a["realized"]), confidence=a.get("confidence"),
            depiction_only=bool(a.get("depiction_only", False)),
            evidence=a.get("evidence"),
            annotator="claude-subagent", protocol="normal-scale single view, no magnification",
        ))

    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(rows)
    real = [r for r in rows if r["realized"]]
    by_bin = defaultdict(lambda: [0, 0])
    for r in rows:
        by_bin[r["plausibility"]][0] += 1
        by_bin[r["plausibility"]][1] += r["realized"]

    # per-item: does at least one of the two samples over-realize?
    per_item = defaultdict(list)
    for r in rows:
        per_item[r["item_id"]].append(r["realized"])
    items_any = sum(any(v) for v in per_item.values())
    items_both = sum(all(v) and len(v) > 1 for v in per_item.values())

    stats = dict(
        n_images=n, n_realized=len(real),
        rate=round(len(real) / n, 4) if n else None,
        by_bin={k: dict(n=v[0], realized=v[1], rate=round(v[1] / v[0], 4) if v[0] else None)
                for k, v in sorted(by_bin.items())},
        n_items=len(per_item), items_with_any_realization=items_any,
        items_with_both_samples_realized=items_both,
        realized_by_target=Counter(r["target"] for r in real).most_common(),
        confidence_of_realized=dict(Counter(r["confidence"] for r in real)),
        depiction_only=sum(r["depiction_only"] for r in rows),
        low_confidence_negatives=sum(
            1 for r in rows if not r["realized"] and r["confidence"] == "low"),
        missing_keys=missing, extra_keys=extra, duplicate_keys=sorted(set(dupes)),
    )
    with open(STATS, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(json.dumps({k: v for k, v in stats.items()
                      if k not in ("realized_by_target", "missing_keys")}, indent=2))
    print(f"\nmissing: {len(missing)}  extra: {len(extra)}  dupes: {len(set(dupes))}")
    print("\nover-realizations:")
    for r in real:
        print(f"  {r['key']:>16}  {r['plausibility'][:4]}  {r['target']:<14} "
              f"[{r['confidence']}]{'[depiction]' if r['depiction_only'] else ''} "
              f"{r['evidence']}")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
