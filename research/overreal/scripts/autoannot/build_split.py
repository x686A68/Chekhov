"""Split the human-labeled images into dev / test for the automated annotator.

Gold images (det_split == gold) are split 50/50 inside each
(family, generator, prompt_cond, wave) stratum, seeded, so every stratum
is represented on both sides. Question wording is iterated on dev only;
kappa is reported on test only. Ambiguous images go to split "amb" for the
side analysis of which annotator the automatic label sides with; excluded
images are dropped.

Writes data/overreal_v1/auto_split.jsonl:
  {image_id, split, family, generator, prompt_cond, wave, gold_labels,
   labels_1, labels_2}
"""
import collections
import json
import random

from common import DS, SPLIT, load_meta

SEED = 20260909


def main():
    meta = load_meta()
    strata = collections.defaultdict(list)
    rows = {}
    for line in open(DS / "det_split.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["split"] == "excluded":
            continue
        m = meta[r["image_id"]]
        if not m.get("file_name"):
            continue
        row = {"image_id": r["image_id"], "split": None, "family": m["family"],
               "generator": m["generator"], "prompt_cond": m.get("prompt_cond"),
               "wave": r["wave"], "gold_labels": r["gold_labels"],
               "labels_1": r["labels_1"], "labels_2": r["labels_2"]}
        rows[r["image_id"]] = row
        if r["split"] == "ambiguous":
            row["split"] = "amb"
        else:
            strata[(m["family"], m["generator"], m.get("prompt_cond"), r["wave"])].append(r["image_id"])

    for key in sorted(strata, key=str):
        ids = sorted(strata[key])
        random.Random(f"{SEED}/{key}").shuffle(ids)
        for i, iid in enumerate(ids):
            rows[iid]["split"] = "dev" if i % 2 == 0 else "test"

    out = sorted(rows.values(), key=lambda r: r["image_id"])
    with open(SPLIT, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    c = collections.Counter((r["family"], r["split"]) for r in out)
    for fam in ["cancellation", "attribution", "figurative", "perspectival"]:
        print(f"{fam:14s} dev={c[(fam,'dev')]:4d} test={c[(fam,'test')]:4d} amb={c[(fam,'amb')]:4d}")
    print(f"{'TOTAL':14s} dev={sum(v for k,v in c.items() if k[1]=='dev'):4d} "
          f"test={sum(v for k,v in c.items() if k[1]=='test'):4d} "
          f"amb={sum(v for k,v in c.items() if k[1]=='amb'):4d} -> {SPLIT}")


if __name__ == "__main__":
    main()
