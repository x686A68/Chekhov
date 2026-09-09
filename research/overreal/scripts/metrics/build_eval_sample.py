"""Build the canonical machine-evaluation sample of overreal_v1.

Cell = (generator, family, prompt_cond); from each cell take
min(100, all) images, seeded, so every machine pass (LLM judge, sampled
metrics, ablations) scores the same fixed set. Marathon rows are excluded
(they have human labels already and no controlled condition).

Output: data/overreal_v1/eval_sample.jsonl
        {image_id, generator, family, prompt_cond}
"""
import collections
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DS = ROOT / "data" / "overreal_v1"
OUT = DS / "eval_sample.jsonl"
PER_CELL = 100
SEED = 20260909


def main():
    cells = collections.defaultdict(list)
    for line in open(DS / "metadata.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if (r.get("file_name") and not r.get("refused")
                and r.get("source_folder") is None):
            cells[(r["generator"], r["family"], r["prompt_cond"])].append(r["image_id"])

    rows = []
    for key in sorted(cells):
        pool = sorted(cells[key])
        rng = random.Random(f"{SEED}/{key}")
        take = pool if len(pool) <= PER_CELL else rng.sample(pool, PER_CELL)
        for iid in sorted(take):
            rows.append({"image_id": iid, "generator": key[0],
                         "family": key[1], "prompt_cond": key[2]})

    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n_cells = len(cells)
    full = sum(1 for k in cells if len(cells[k]) <= PER_CELL)
    print(f"{len(rows)} images across {n_cells} cells "
          f"({full} cells taken in full) -> {OUT}")


if __name__ == "__main__":
    main()
