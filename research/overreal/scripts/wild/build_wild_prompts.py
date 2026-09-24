"""Build data/generation/wild_prompts.jsonl for the generation condition "wild".

Inputs (produced by merge_labels.py):
  data/wild/positives.jsonl            random-sample positives (base-rate sample)
  data/wild/targeted/positives.jsonl   trigger-word candidates (mental, perspectival top-up)

Only high/medium positives are in those files. Families are mapped to the
pipeline keys used by the generators and the automatic annotator:
  existence -> cancellation, mental -> attribution, figurative, perspectival.

Output rows: {item_id, family, prompt, target, wid, source, sample, cue, confidence}
  item_id = "<family>/wild_NNNN" so image_path() yields <family>_wild_NNNN__s<seed>.png
  sample  = "random" | "targeted"
Targeted rows are only added for families the flag --topup names (default:
attribution,perspectival), so the base-rate families stay purely random.
Existing item_ids are kept stable across re-runs (order: random first, then
targeted, each in wid order).
"""
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
W = ROOT / "data" / "wild"
OUT = ROOT / "data" / "generation" / "wild_prompts.jsonl"
FAM = {"existence": "cancellation", "mental": "attribution",
       "figurative": "figurative", "perspectival": "perspectival"}


def read(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")] if p.exists() else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topup", default="attribution,perspectival")
    args = ap.parse_args()
    topup = set(args.topup.split(",")) if args.topup else set()

    rows = []
    seen = set()
    per_target = {}
    dropped = {"pretend": 0, "dup_prompt": 0, "target_cap": 0}
    for sample, path in (("random", W / "positives.jsonl"),
                         ("targeted", W / "targeted" / "positives.jsonl")):
        for r in sorted(read(path), key=lambda r: r["wid"]):
            fam = FAM[r["family"]]
            if sample == "targeted" and fam not in topup:
                continue
            # Ruling (2026-09-24): "pretending to be X" is role play, not a mental state.
            if sample == "targeted" and "pretend" in r["cue"].lower():
                dropped["pretend"] += 1
                continue
            # (targeted rows only, so the random part keeps its item_ids)
            # DiffusionDB holds many variants of one user's prompt: same opening, same
            # target. Dedupe on the normalised first 60 characters, then cap each
            # (family, target) at 2 prompts so no single user dominates a family.
            key = (re.sub(r"[^a-z0-9 ]", "", r["prompt"].lower())[:60].strip()
                   if sample == "targeted" else r["prompt"].strip().lower())
            if key in seen:
                dropped["dup_prompt"] += 1
                continue
            tkey = (fam, re.sub(r"[^a-z0-9 ]", "", r["target"].lower()).strip())
            if sample == "targeted" and per_target.get(tkey, 0) >= 2:
                dropped["target_cap"] += 1
                continue
            seen.add(key)
            per_target[tkey] = per_target.get(tkey, 0) + 1
            rows.append({"family": fam, "prompt": r["prompt"], "target": r["target"],
                         "wid": r["wid"], "source": r["source"], "sample": sample,
                         "cue": r["cue"], "confidence": r["confidence"]})
    print("dropped:", dropped)

    counts = {}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            r = {"item_id": f"{r['family']}/wild_{i:04d}", **r}
            counts[(r["family"], r["sample"])] = counts.get((r["family"], r["sample"]), 0) + 1
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} items to {OUT}")
    for k in sorted(counts):
        print(f"  {k[0]:13s} {k[1]:9s} {counts[k]}")


if __name__ == "__main__":
    main()
