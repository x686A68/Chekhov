"""Outcome shares under the explicit-constraint prompts (reviewer KXAW, Q2).

Inputs
  data/generation/constraint_prompts.jsonl                         items
  data/overreal_v1/auto/final__claude-opus-5-api__constraint_s*.jsonl  judge output
  data/overreal_v1/auto/final__claude-opus-5-api__sample.jsonl     judge output on the Gen eval sample (raw prompts)
  data/overreal_v1/eval_sample.jsonl

Two lines of outcomes, with "other" kept in the denominator:
  target appears : disruptive / silent / integrated
  target absent  : withheld / other
Baseline = raw-prompt images of the same items in the eval sample (both seeds).
Output: research/overreal/constraint/rates.md
"""
import collections
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
GEN = ROOT / "data" / "generation"
DS = ROOT / "data" / "overreal_v1"
OUT = ROOT / "research" / "overreal" / "constraint" / "rates.md"
MODELS = ["flux", "qwen-image", "omnigen2", "sd35m", "sd35l"]
NAME = {"flux": "FLUX.1-dev", "qwen-image": "Qwen-Image", "omnigen2": "OmniGen2",
        "sd35m": "SD3.5-Medium", "sd35l": "SD3.5-Large"}
GEN_KEY = {"flux.1-dev": "flux", "qwen-image": "qwen-image", "omnigen2": "omnigen2",
           "sd3.5-medium": "sd35m", "sd3.5-large": "sd35l"}
FAMS = ["attribution", "figurative"]
FAM_NAME = {"attribution": "Mental-state", "figurative": "Figurative"}
LABS = ["disruptive", "silent", "integrated", "withheld", "other"]


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def main():
    items = {r["item_id"] for r in read_jsonl(GEN / "constraint_prompts.jsonl")}
    fam_of = {r["item_id"]: r["family"] for r in read_jsonl(GEN / "constraint_prompts.jsonl")}

    cons = collections.defaultdict(list)           # (model, family) -> labels
    for f in glob.glob(str(DS / "auto" / "final__claude-opus-5-api__constraint_s*.jsonl")):
        for r in read_jsonl(f):
            if not r.get("label"):
                continue
            _, model, _, base = r["image_id"].split("/")   # gen/<model>/constraint/<fam>_<nnnn>__s0.png
            fam, num = base.split("__")[0].rsplit("_", 1)
            cons[(model, fam)].append(r["label"])

    base = collections.defaultdict(list)
    es = {r["image_id"]: r for r in read_jsonl(DS / "eval_sample.jsonl")}
    for r in read_jsonl(DS / "auto" / "final__claude-opus-5-api__sample.jsonl"):
        e = es.get(r["image_id"])
        if not e or e["prompt_cond"] != "raw" or not r.get("label"):
            continue
        item = "/".join(r["image_id"].split("/")[:2])
        if item in items and e["generator"] in GEN_KEY:
            base[(GEN_KEY[e["generator"]], fam_of[item])].append(r["label"])

    def row(labels):
        n = len(labels)
        c = collections.Counter(labels)
        s = {l: c[l] / n for l in LABS} if n else {l: float("nan") for l in LABS}
        s["present"] = s["disruptive"] + s["silent"] + s["integrated"]
        s["n"] = n
        return s

    lines = ["# Explicit-constraint prompts: outcome shares (other kept in the denominator)", "",
             "Constraint = original prompt + \"No <target> should appear in the image.\", seed 0.",
             "Baseline = raw prompts of the same items in the Gen eval sample (both seeds).", ""]
    for fam in FAMS:
        lines += [f"## {FAM_NAME[fam]}", "",
                  "| Model | Prompts | n | target present | DO | SO | I | target absent | W | other |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for m in MODELS:
            for tag, src in (("raw", base), ("constraint", cons)):
                s = row(src[(m, fam)])
                lines.append(f"| {NAME[m]} | {tag} | {s['n']} | {s['present']:.2f} | {s['disruptive']:.2f} | "
                             f"{s['silent']:.2f} | {s['integrated']:.2f} | {s['withheld'] + s['other']:.2f} | "
                             f"{s['withheld']:.2f} | {s['other']:.2f} |")
        lines.append("")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
