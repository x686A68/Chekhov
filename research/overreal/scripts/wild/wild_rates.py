"""Over-realization rates on the in-the-wild prompt set, next to the OverReal-Gen rates.

Inputs
  data/generation/wild_prompts.jsonl                     items (family, sample = random | targeted)
  data/overreal_v1/auto/final__claude-opus-5-api__wild_s*.jsonl   Opus judge output (run_auto.py --manifests)
  data/generation/wild_refused_gpt-image.txt              GPT-Image items blocked by moderation
  data/overreal_v1/auto/final__claude-opus-5-api__sample.jsonl    Opus judge on the OverReal-Gen eval sample (Table 2b)
  data/overreal_v1/eval_sample.jsonl, metadata.jsonl

Outputs
  research/overreal/wild/rates.md      per family x system: n, OR, DO, SO, I, W on wild and on Gen
  research/overreal/wild/rates.tex     LaTeX rows for the paper table
Rates are over judged images (label != other); refusals are excluded from the denominator.
"""
import collections
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "research" / "overreal" / "scripts" / "autoannot"))
from common import derive_label  # noqa: E402

GEN = ROOT / "data" / "generation"
DS = ROOT / "data" / "overreal_v1"
OUTDIR = ROOT / "research" / "overreal" / "wild"
FAMS = ["cancellation", "attribution", "figurative", "perspectival"]
FAM_NAME = {"cancellation": "Existence-canceling", "attribution": "Mental-state",
            "figurative": "Figurative", "perspectival": "Perspectival"}
# manifest model key -> paper name; metadata generator key -> paper name
MODEL_NAME = {"gpt-image": "GPT-Image", "nanobanana": "Nano Banana", "ideogram": "Ideogram",
              "flux": "FLUX.1-dev", "qwen-image": "Qwen-Image", "omnigen2": "OmniGen2",
              "sd35m": "SD3.5-Medium", "sd35l": "SD3.5-Large"}
GEN_KEY = {"gpt-image-1.5": "gpt-image", "gemini-2.5-flash-image-api": "nanobanana",
           "ideogram-v3": "ideogram", "flux.1-dev": "flux", "qwen-image": "qwen-image",
           "omnigen2": "omnigen2", "sd3.5-medium": "sd35m", "sd3.5-large": "sd35l"}
ORDER = ["gpt-image", "nanobanana", "ideogram", "flux", "qwen-image", "omnigen2", "sd35m", "sd35l"]
OUTS = ["disruptive", "silent", "integrated", "withheld"]


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def rates(labels):
    """labels: list of outcome labels (other excluded). -> dict of shares, or None if < 10."""
    n = len(labels)
    if n < 10:
        return None
    c = collections.Counter(labels)
    d = {o: c[o] / n for o in OUTS}
    d["or"] = d["disruptive"] + d["silent"]
    d["n"] = n
    return d


def wild_labels():
    items = {r["item_id"]: r for r in read_jsonl(GEN / "wild_prompts.jsonl")}
    cells = collections.defaultdict(list)          # (model, family, sample) -> labels
    seen = set()                                    # shard files share pre-seeded rows
    for f in sorted(glob.glob(str(DS / "auto" / "final__claude-opus-5-api__wild_*.jsonl"))):
        for r in read_jsonl(f):
            if r["image_id"] in seen:
                continue
            seen.add(r["image_id"])
            lab = derive_label(r["answers"], r["questions"])
            if not lab or lab == "other":
                continue
            _, model, _, name = r["image_id"].split("/")
            fam, rest = name.split("_wild_")
            item = f"{fam}/wild_{rest.split('__')[0]}"
            if item not in items:
                continue
            cells[(model, fam, items[item]["sample"])].append(lab)
    return cells, items


def gen_labels():
    meta = {}
    for r in read_jsonl(DS / "metadata.jsonl"):
        if r.get("prompt_cond") in ("raw", "deployed") and r.get("file_name") and not r.get("refused"):
            meta[r["image_id"]] = (GEN_KEY.get(r["generator"]), r["family"])
    sample = {r["image_id"] for r in read_jsonl(DS / "eval_sample.jsonl")}
    cells = collections.defaultdict(list)
    for r in read_jsonl(DS / "auto" / "final__claude-opus-5-api__sample.jsonl"):
        lab = derive_label(r["answers"], r["questions"])
        if not lab or lab == "other" or r["image_id"] not in sample or r["image_id"] not in meta:
            continue
        model, fam = meta[r["image_id"]]
        if model:
            cells[(model, fam)].append(lab)
    return cells


def fmt(d, key):
    return "---" if d is None else f"{d[key]:.2f}"


def main():
    wild, items = wild_labels()
    gen = gen_labels()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    n_items = collections.Counter((r["family"], r["sample"]) for r in items.values())

    md = ["# Over-realization in the wild (Opus judge, protocol final)", "",
          "Wild items: " + ", ".join(f"{FAM_NAME[f]} {n_items[(f, 'random')]} random + "
                                     f"{n_items[(f, 'targeted')]} targeted" for f in FAMS), "",
          "Rates over judged images (label != other). OR = DO + SO. Gen = OverReal-Gen eval sample, same judge.", ""]
    tex = []
    for fam in FAMS:
        md += [f"## {FAM_NAME[fam]}", "",
               "| system | wild n | wild OR | DO | SO | I | W | Gen n | Gen OR | Gen DO | Gen SO |",
               "|---|---|---|---|---|---|---|---|---|---|---|"]
        tex.append(f"% {FAM_NAME[fam]}")
        for m in ORDER:
            labs = wild.get((m, fam, "random"), []) + wild.get((m, fam, "targeted"), [])
            w = rates(labs)
            g = rates(gen.get((m, fam), []))
            md.append(f"| {MODEL_NAME[m]} | {w['n'] if w else len(labs)} | {fmt(w,'or')} | {fmt(w,'disruptive')} | "
                      f"{fmt(w,'silent')} | {fmt(w,'integrated')} | {fmt(w,'withheld')} | "
                      f"{g['n'] if g else 0} | {fmt(g,'or')} | {fmt(g,'disruptive')} | {fmt(g,'silent')} |")
            tex.append(f"{MODEL_NAME[m]:13s} & {fmt(w,'disruptive')} & {fmt(w,'silent')} & {fmt(w,'integrated')} "
                       f"& {fmt(w,'withheld')} & {fmt(g,'or')} \\\\")
        md.append("")
        # pooled over systems, random-sample items only (the base-rate sample)
        rnd = [l for m in ORDER for l in wild.get((m, fam, "random"), [])]
        allw = [l for m in ORDER for l in wild.get((m, fam, "random"), []) + wild.get((m, fam, "targeted"), [])]
        r1, r2 = rates(rnd), rates(allw)
        md.append(f"Pooled over systems: random-sample items OR = {fmt(r1,'or')} (n={len(rnd)}); "
                  f"all wild items OR = {fmt(r2,'or')} (n={len(allw)}).")
        md.append("")
    (OUTDIR / "rates.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (OUTDIR / "rates.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
