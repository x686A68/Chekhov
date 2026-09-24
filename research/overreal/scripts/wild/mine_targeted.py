"""Targeted mining for the two rare families (mental, perspectival).

The random 10k sample (sample_prompts.py) gives the base rate. It yields too few
mental-state and perspectival prompts for a conditional failure rate, so this
script pulls trigger-word candidates from the full DiffusionDB and Pick-a-Pic
pools (same cleaning filters as the random sample), excluding prompts already
in the random sample. Candidates still go through the rubric classification;
the trigger words only raise the density. Selection remains on prompt form.

Output: data/wild/targeted/chunks/t_XX.jsonl, 250 prompts per chunk, fields
  {wid, source, source_id, prompt, cand_family, cand_cue}
wid prefix "t" keeps them apart from the random sample ("w").
"""
import collections
import json
import re
from pathlib import Path

from sample_prompts import load_diffusiondb, load_pickapic, dedupe

ROOT = Path(__file__).resolve().parents[4]
W = ROOT / "data" / "wild"
OUT = W / "targeted" / "chunks"

PATS = {
    "mental": re.compile(
        r"\b(dreaming of|dreams of|dreaming about|dreams about|imagining|imagines|"
        r"remembering|remembers|believes|believing|thinking of|thinking about|"
        r"daydreaming|fantasizing|fantasising|wishing for|hoping for|longing for|"
        r"pretending|hallucinating|picturing)\b", re.I),
    "perspectival": re.compile(
        r"\b(hidden inside|hidden behind|hidden under|hidden underneath|hiding inside|"
        r"hiding behind|hiding under|hiding underneath|inside a closed|inside a sealed|"
        r"sealed inside|wrapped inside|locked inside|in his pocket|in her pocket|"
        r"in their pocket|in a closed|behind a closed|behind the door|behind a door|"
        r"behind a wall|behind the wall|under the blanket|under a blanket|"
        r"under the bed|facing away|back to the camera|back to camera|out of sight|"
        r"can't be seen|cannot be seen|not visible|out of view|turned away)\b", re.I),
}


def main():
    sample = {json.loads(l)["prompt"].lower() for l in open(W / "sample_10k.jsonl", encoding="utf-8")}
    pool = ([("diffusiondb", sid, p) for sid, p in dedupe(load_diffusiondb())]
            + [("pickapic", sid, p) for sid, p in dedupe(load_pickapic())])
    rows = []
    stats = collections.Counter()
    for src, sid, p in pool:
        if p.lower() in sample:
            continue
        for fam, rx in PATS.items():
            m = rx.search(p)
            if m:
                rows.append({"source": src, "source_id": sid, "prompt": p,
                             "cand_family": fam, "cand_cue": m.group(0).lower()})
                stats[(fam, src)] += 1
                break
    print("candidates:", len(rows), dict(stats))
    OUT.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("t_*.jsonl"):
        f.unlink()
    for i, r in enumerate(rows):
        r["wid"] = f"t{i:05d}"
    for i in range(0, len(rows), 250):
        with open(OUT / f"t_{i // 250:02d}.jsonl", "w", encoding="utf-8") as f:
            for r in rows[i:i + 250]:
                f.write(json.dumps({k: r[k] for k in ("wid", "source", "source_id", "prompt",
                                                      "cand_family", "cand_cue")},
                                   ensure_ascii=False) + "\n")
    print("chunks:", (len(rows) + 249) // 250)


if __name__ == "__main__":
    main()
