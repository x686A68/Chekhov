"""Sample real user prompts for the in-the-wild over-realization study.

Sources (data/wild/source/):
  metadata.parquet                    DiffusionDB 2M metadata (poloclub/diffusiondb)
  pickapic/data/train-*.parquet       Pick-a-Pic rankings (yuvalkirstain/PickaPic-rankings)

Filters, applied before sampling:
  - normalise whitespace; drop duplicates (case-insensitive, whitespace-collapsed)
  - DiffusionDB: prompt_nsfw < 0.5 and image_nsfw < 0.5
  - English only: >= 90% ASCII characters
  - 3 to 120 words

Output: data/wild/sample_10k.jsonl, one line per prompt:
  {wid, source, source_id, prompt}
Seeded, so the sample is reproducible.
"""
import argparse
import glob
import json
import random
import re
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "data" / "wild" / "source"
OUT = ROOT / "data" / "wild" / "sample_10k.jsonl"


def norm(p):
    return re.sub(r"\s+", " ", str(p)).strip()


def keep(p):
    if not p:
        return False
    n = len(p.split())
    if n < 3 or n > 120:
        return False
    ascii_share = sum(ord(c) < 128 for c in p) / len(p)
    return ascii_share >= 0.9


def load_diffusiondb():
    t = pq.read_table(SRC / "metadata.parquet",
                      columns=["image_name", "prompt", "prompt_nsfw", "image_nsfw"])
    rows = []
    for name, p, pn, im in zip(t["image_name"].to_pylist(), t["prompt"].to_pylist(),
                               t["prompt_nsfw"].to_pylist(), t["image_nsfw"].to_pylist()):
        if pn is not None and pn >= 0.5:
            continue
        if im is not None and im >= 0.5:
            continue
        p = norm(p)
        if keep(p):
            rows.append((name, p))
    return rows


def load_pickapic():
    rows = []
    for f in glob.glob(str(SRC / "pickapic" / "data" / "*.parquet")):
        t = pq.read_table(f, columns=["ranking_id", "prompt"])
        for rid, p in zip(t["ranking_id"].to_pylist(), t["prompt"].to_pylist()):
            p = norm(p)
            if keep(p):
                rows.append((str(rid), p))
    return rows


def dedupe(rows):
    seen, out = set(), []
    for sid, p in rows:
        k = p.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append((sid, p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-diffusiondb", type=int, default=7000)
    ap.add_argument("--n-pickapic", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=20260924)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    ddb = dedupe(load_diffusiondb())
    pap = dedupe(load_pickapic())
    print(f"diffusiondb: {len(ddb)} unique prompts after filters")
    print(f"pickapic:    {len(pap)} unique prompts after filters")

    sample = ([("diffusiondb", sid, p) for sid, p in rng.sample(ddb, args.n_diffusiondb)]
              + [("pickapic", sid, p) for sid, p in rng.sample(pap, args.n_pickapic)])
    rng.shuffle(sample)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for i, (src, sid, p) in enumerate(sample):
            f.write(json.dumps({"wid": f"w{i:05d}", "source": src, "source_id": sid,
                                "prompt": p}, ensure_ascii=False) + "\n")
    print(f"wrote {len(sample)} prompts to {OUT}")


if __name__ == "__main__":
    main()
