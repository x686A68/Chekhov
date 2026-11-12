"""Wild human check: per-family confusion (automatic vs. human) and the
over-realization rate after reweighting the automatic labels by the human
ones (reviewer 3P6s, Q2). Run from the repo root with the Fraud env python.
Output: research/overreal/wild/human_check_reweight.md
"""
import collections, glob, json, os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
sys.path.insert(0, "research/overreal/scripts/autoannot")
from common import derive_label
rj = lambda p: [json.loads(l) for l in open(p)]
samp = {r["image_id"]: r for r in rj("research/overreal/wild/human_check_sample.jsonl")}
ann = {}
for f in sorted(glob.glob("data/wild_check_annotations/ann-*.jsonl")):
    for r in rj(f):
        ann[r["image_id"]] = r                      # latest record wins
FAM = {"cancellation": "Existence-canceling", "attribution": "Mental-state",
       "figurative": "Figurative", "perspectival": "Perspectival"}
LABS = ["disruptive", "silent", "integrated", "withheld", "other"]
conf = collections.defaultdict(collections.Counter)
for iid, s in samp.items():
    a = ann[iid]; j = s["judge_label"]
    h = j if j in a["labels"] else a["labels"][0]  # two-label record credits the judge if it matches
    conf[s["family"]][(j, h)] += 1
uni = collections.defaultdict(collections.Counter); seen = set()
for f in sorted(glob.glob("data/overreal_v1/auto/final__claude-opus-5-api__wild_*.jsonl")):
    for r in rj(f):
        if r["image_id"] in seen: continue
        seen.add(r["image_id"])
        fam = r["image_id"].split("/")[-1].split("_wild_")[0]
        l = r.get("label") or derive_label(r["answers"], r["questions"])
        if l: uni[fam][l] += 1
OR = lambda l: l in ("disruptive", "silent")
out = ["# Wild human check: confusion per family and reweighted over-realization rate", ""]
for fam in FAM:
    c = conf[fam]
    out += [f"## {FAM[fam]} (sample n={sum(c.values())})", "",
            "| automatic \\ human | " + " | ".join(LABS) + " |", "|---" * (len(LABS) + 1) + "|"]
    for j in LABS:
        row = [c[(j, h)] for h in LABS]
        if sum(row): out.append(f"| {j} | " + " | ".join(str(x) for x in row) + " |")
    u = uni[fam]; N = sum(v for k, v in u.items() if k != "other")
    judge_or = sum(v for k, v in u.items() if OR(k)) / N
    corr = sum(u[j] * sum(c[(j, h)] for h in LABS if OR(h)) / sum(c[(j, h)] for h in LABS)
               for j in LABS if sum(c[(j, h)] for h in LABS)) / N
    out += ["", f"Universe N={N} (other excluded); automatic OR rate {judge_or:.3f}; human-reweighted OR rate {corr:.3f}", ""]
open("research/overreal/wild/human_check_reweight.md", "w").write("\n".join(out)); print("\n".join(out))
