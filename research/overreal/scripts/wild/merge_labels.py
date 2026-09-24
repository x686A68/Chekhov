"""Merge the per-chunk classification outputs of the wild-prompt study.

Reads data/wild/chunks/chunk_*.jsonl (inputs) and data/wild/out/chunk_*.out.jsonl
(labels), checks coverage, and writes:
  data/wild/labels.jsonl        every label line joined with source and prompt
  data/wild/positives.jsonl     family != none and confidence in {high, medium}
  data/wild/summary.md          counts per family x confidence, per flag, per source
"""
import collections
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
W = ROOT / "data" / "wild"
FAMS = ["existence", "mental", "figurative", "perspectival"]


def main():
    prompts = {}
    for f in sorted(glob.glob(str(W / "chunks" / "chunk_*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            prompts[r["wid"]] = r
    labels = []
    seen = collections.Counter()
    bad = []
    for f in sorted(glob.glob(str(W / "out" / "chunk_*.out.jsonl"))):
        for n, line in enumerate(open(f, encoding="utf-8"), 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                bad.append((f, n))
                continue
            if r.get("wid") not in prompts:
                bad.append((f, n, "unknown wid"))
                continue
            seen[r["wid"]] += 1
            r["family"] = r.get("family", "none") or "none"
            r["confidence"] = r.get("confidence", "") or ""
            r["flag"] = r.get("flag", "") or ""
            labels.append(r)

    missing = [w for w in prompts if seen[w] == 0]
    dups = {w: c for w, c in seen.items() if c > 1}
    print(f"prompts {len(prompts)}, label lines {len(labels)}, "
          f"missing {len(missing)}, multi-line wids {len(dups)}, bad lines {len(bad)}")
    if missing:
        print("missing wids (first 20):", missing[:20])
    if bad:
        print("bad lines:", bad[:20])

    with open(W / "labels.jsonl", "w", encoding="utf-8") as f:
        for r in labels:
            p = prompts[r["wid"]]
            f.write(json.dumps({**r, "source": p["source"], "source_id": p["source_id"],
                                "prompt": p["prompt"]}, ensure_ascii=False) + "\n")

    pos = [r for r in labels if r["family"] in FAMS and r["confidence"] in ("high", "medium")]
    with open(W / "positives.jsonl", "w", encoding="utf-8") as f:
        for r in pos:
            p = prompts[r["wid"]]
            f.write(json.dumps({"wid": r["wid"], "source": p["source"], "family": r["family"],
                                "target": r["target"], "cue": r["cue"],
                                "confidence": r["confidence"], "prompt": p["prompt"]},
                               ensure_ascii=False) + "\n")

    fam_conf = collections.Counter((r["family"], r["confidence"]) for r in labels if r["family"] in FAMS)
    flags = collections.Counter(r["flag"] for r in labels if r["flag"])
    pos_src = collections.Counter((r["family"], prompts[r["wid"]]["source"]) for r in pos)
    n_src = collections.Counter(p["source"] for p in prompts.values())
    pos_wids = {r["wid"] for r in pos}

    lines = ["# Wild-prompt study: classification summary", "",
             f"Prompts classified: {len(prompts)} "
             f"(diffusiondb {n_src['diffusiondb']}, pickapic {n_src['pickapic']}); "
             f"label lines {len(labels)}; missing {len(missing)}.", "",
             "## Positives by family and confidence", "",
             "| family | high | medium | low | kept (high+medium) |", "|---|---|---|---|---|"]
    for fm in FAMS:
        h, m, l = fam_conf[(fm, "high")], fam_conf[(fm, "medium")], fam_conf[(fm, "low")]
        lines.append(f"| {fm} | {h} | {m} | {l} | {h + m} |")
    tot_kept = sum(fam_conf[(fm, c)] for fm in FAMS for c in ("high", "medium"))
    lines += ["", f"Kept positives: {tot_kept} label lines on {len(pos_wids)} prompts "
                  f"({100 * len(pos_wids) / len(prompts):.2f}% of the sample).", "",
              "## Kept positives by source", "",
              "| family | diffusiondb | pickapic |", "|---|---|---|"]
    for fm in FAMS:
        lines.append(f"| {fm} | {pos_src[(fm, 'diffusiondb')]} | {pos_src[(fm, 'pickapic')]} |")
    lines += ["", "## Rejected near-misses by flag", "", "| flag | count |", "|---|---|"]
    for k, v in flags.most_common():
        lines.append(f"| {k} | {v} |")
    (W / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
