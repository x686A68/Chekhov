"""Stratified sample of in-the-wild judge predictions for a blind human check
(reviewer 3P6s, Q2), plus the staging dir for the annotation Space.

Universe: the 397 wild prompts x 8 systems, seed 0 (the images behind
Table 7b), with the final Opus judge label. Strata: family x judge label.
Allocation per family: proportional to the judge-label counts with a floor
(labels with fewer than MIN_STRATUM images are dropped).

Writes
  research/overreal/wild/human_check_sample.jsonl   image_id, family, generator,
                                                    judge_label, prompt, target
  data/wild_check_staging/images/<sha1(image_id)[:16]>.png   (hardlinks)
  data/wild_check_staging/tasks.jsonl                image_id, file, family,
                                                    prompt, target  (no label)
"""
import collections
import glob
import hashlib
import json
import os
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
GEN = ROOT / "data" / "generation"
DS = ROOT / "data" / "overreal_v1"
OUT = ROOT / "research" / "overreal" / "wild" / "human_check_sample.jsonl"
STAGING = ROOT / "data" / "wild_check_staging"
TARGET = {"perspectival": 100, "attribution": 34, "figurative": 34, "cancellation": 34}
FLOOR, MIN_STRATUM = 15, 5
SEED = 0


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main():
    items = {r["item_id"]: r for r in read_jsonl(GEN / "wild_prompts.jsonl")}
    labels = {}
    for f in sorted(glob.glob(str(DS / "auto" / "final__claude-opus-5-api__wild_*.jsonl"))):
        for r in read_jsonl(f):
            if r.get("label"):
                labels[r["image_id"]] = r["label"]
    universe = []
    for iid, lab in labels.items():
        _, model, _, base = iid.split("/")
        stem, seed = base[:-4].split("__")
        fam, num = stem.split("_wild_")
        item = f"{fam}/wild_{num}"
        if seed != "s0" or item not in items:
            continue
        universe.append({"image_id": iid, "family": fam, "generator": model,
                         "judge_label": lab, "prompt": items[item]["prompt"],
                         "target": items[item]["target"], "item_id": item})
    by = collections.defaultdict(list)
    for r in universe:
        by[(r["family"], r["judge_label"])].append(r)
    print(f"universe: {len(universe)} images "
          f"({collections.Counter(r['generator'] for r in universe)})")

    rng = random.Random(SEED)
    sample = []
    for fam, n_fam in TARGET.items():
        strata = {k: v for k, v in by.items() if k[0] == fam and len(v) >= MIN_STRATUM}
        tot = sum(len(v) for v in strata.values())
        alloc = {k: max(min(FLOOR, len(v)), round(n_fam * len(v) / tot)) for k, v in strata.items()}
        # trim the largest stratum so the family hits its target
        while sum(alloc.values()) > n_fam:
            k = max(alloc, key=alloc.get)
            alloc[k] -= 1
        for k, n in alloc.items():
            pool = sorted(strata[k], key=lambda r: r["image_id"])
            rng.shuffle(pool)
            sample += pool[:n]
            print(f"  {fam:13s} {k[1]:11s} {n:3d} / {len(strata[k])}")
    rng.shuffle(sample)
    print(f"sample: {len(sample)} images; generators "
          f"{dict(collections.Counter(r['generator'] for r in sample))}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in sample:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    (STAGING / "images").mkdir(parents=True, exist_ok=True)
    with open(STAGING / "tasks.jsonl", "w", encoding="utf-8") as out:
        for r in sample:
            _, model, _, base = r["image_id"].split("/")
            src = GEN / "images" / model / "wild" / base
            assert src.exists(), src
            h = hashlib.sha1(r["image_id"].encode()).hexdigest()[:16]
            dst = STAGING / "images" / f"{h}.png"
            if not dst.exists():
                os.link(src, dst)
            out.write(json.dumps({"image_id": r["image_id"], "file": f"images/{h}.png",
                                  "family": r["family"], "prompt": r["prompt"],
                                  "target": r["target"]}, ensure_ascii=False) + "\n")
    print(f"staged -> {STAGING}")


if __name__ == "__main__":
    main()
