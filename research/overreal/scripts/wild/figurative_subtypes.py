"""Figurative prompts split by hand into appearance-based comparisons (shape,
colour, texture, size, visual resemblance) and quality-based ones (a
non-visual quality, behaviour or manner, including idioms), on OverReal-Gen
and on the in-the-wild set. Per subtype: outcome shares, per-generator
over-realization and disruptive rates with prompt-level bootstrap intervals,
and rank correlations between subsets (reviewer KXAW, follow-up).
Run from the repo root with the Fraud env python.
"""
import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
import collections, glob, random, sys
sys.path.insert(0, 'research/overreal/scripts/wild'); sys.path.insert(0, 'research/overreal/scripts/autoannot')
import wild_rates as W
from common import derive_label

# hand classification, by position in the figurative part of each prompt file
GEN_APPEAR = set([31, 32, 34, 36, 37, 38, 39, 40, 41, 42, 43, 44] + list(range(46, 90))
                 + [92, 93, 94, 104, 169, 171, 173, 213, 221] + list(range(223, 233)) + [234, 236])
WILD_QUALITY = {5, 6, 25, 26, 45, 53, 90, 92, 100}
GENS = [("gpt-image-1.5", "GPT"), ("gemini-2.5-flash-image-api", "NB"), ("ideogram-v3", "Ideo"),
        ("flux.1-dev", "FLUX"), ("qwen-image", "Qwen"), ("omnigen2", "Omni"),
        ("sd3.5-medium", "SD-M"), ("sd3.5-large", "SD-L")]

gen = [r for r in W.read_jsonl('data/generation/prompts.jsonl') if r['family'] == 'figurative']
wild = [r for r in W.read_jsonl('data/generation/wild_prompts.jsonl') if r['family'] == 'figurative']
gtype = {r['item_id']: ('appearance' if i in GEN_APPEAR else 'quality') for i, r in enumerate(gen)}
wtype = {r['item_id']: ('quality' if i in WILD_QUALITY else 'appearance') for i, r in enumerate(wild)}
for name, t in (("OverReal-Gen", gtype), ("in the wild", wtype)):
    c = collections.Counter(t.values()); n = len(t)
    print(f"{name}: appearance {c['appearance']}/{n} = {c['appearance']/n:.0%}, quality {c['quality']}/{n} = {c['quality']/n:.0%}")

# rows: (set, subtype, generator, prompt, label)
rows = []
meta = {}
for r in W.read_jsonl('data/overreal_v1/metadata.jsonl'):
    if (r.get("file_name") and not r.get("refused") and r.get("source_folder") is None
            and r.get("prompt_cond") in ("raw", "deployed") and r["family"] == "figurative"):
        meta[r["image_id"]] = r["generator"]
sample = {r["image_id"] for r in W.read_jsonl('data/overreal_v1/eval_sample.jsonl')}
for r in W.read_jsonl('data/overreal_v1/auto/final__claude-opus-5-api__sample.jsonl'):
    if r["image_id"] not in sample or r["image_id"] not in meta:
        continue
    item = '/'.join(r['image_id'].split('/')[:2]); l = derive_label(r['answers'], r['questions'])
    if l and l != 'other' and item in gtype:
        rows.append(("gen", gtype[item], meta[r["image_id"]], item, l))
KEY2GEN = {v: k for k, v in W.GEN_KEY.items()}
seen = set()
for f in sorted(glob.glob('data/overreal_v1/auto/final__claude-opus-5-api__wild_*.jsonl')):
    for r in W.read_jsonl(f):
        if r['image_id'] in seen:
            continue
        seen.add(r['image_id']); _, m, _, name = r['image_id'].split('/')
        if not name.startswith('figurative'):
            continue
        fam, rest = name.split('_wild_'); item = f"{fam}/wild_{rest.split('__')[0]}"
        l = derive_label(r['answers'], r['questions'])
        if l and l != 'other' and item in wtype and m in KEY2GEN:
            rows.append(("wild", wtype[item], KEY2GEN[m], item, l))

def shares(labs):
    c = collections.Counter(labs); n = len(labs)
    do, so = c['disruptive'] / n, c['silent'] / n
    return dict(n=n, DO=do, SO=so, I=c['integrated'] / n, W=c['withheld'] / n, OR=do + so,
                share=do / (do + so) if do + so else float('nan'))

print("\nOutcome shares pooled over generators")
for s in ("gen", "wild"):
    for t in ("appearance", "quality", None):
        labs = [r[4] for r in rows if r[0] == s and (t is None or r[1] == t)]
        x = shares(labs)
        print(f"  {s:4s} {str(t or 'all'):10s} n={x['n']:4d} DO {x['DO']:.2f} SO {x['SO']:.2f} I {x['I']:.2f} W {x['W']:.2f} | OR {x['OR']:.2f} DO/OR {x['share']:.2f}")

def boot(sub, f, B=2000, seed=0):
    by = collections.defaultdict(list)
    for r in sub:
        by[r[3]].append(r[4])
    keys = list(by); rng = random.Random(seed); vals = []
    for _ in range(B):
        labs = [l for k in rng.choices(keys, k=len(keys)) for l in by[k]]
        vals.append(f(labs))
    vals.sort(); return vals[int(.025 * B)], vals[int(.975 * B)]

def spearman(a, b):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0] * len(v); i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            for k in range(i, j + 1):
                r[o[k]] = (i + j) / 2
            i = j + 1
        return r
    ra, rb = rank(a), rank(b); n = len(a); ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** .5
    return num / den

print("\nPer generator: over-realization rate [95% CI] / disruptive rate, n images")
tab = {}
for s, t in (("gen", "appearance"), ("gen", "quality"), ("gen", None), ("wild", "appearance"), ("wild", "quality"), ("wild", None)):
    print(f"  {s} {t or 'all'}")
    for g, nm in GENS:
        sub = [r for r in rows if r[0] == s and (t is None or r[1] == t) and r[2] == g]
        x = shares([r[4] for r in sub]); lo, hi = boot(sub, lambda L: sum(l in ('disruptive', 'silent') for l in L) / len(L))
        tab[(s, t, nm)] = x
        print(f"    {nm:5s} n={x['n']:3d} OR {x['OR']:.2f} [{lo:.2f},{hi:.2f}]  DO {x['DO']:.2f}  SO {x['SO']:.2f}  SO>DO {x['SO'] > x['DO']}")
names = [nm for _, nm in GENS]
def col(s, t, k):
    return [tab[(s, t, nm)][k] for nm in names]
print("\nSpearman between generator rankings (over-realization rate / disruptive rate)")
for a, b in ((("gen", "quality"), ("gen", None)), (("gen", "appearance"), ("gen", None)),
             (("gen", "appearance"), ("gen", "quality")), (("gen", "appearance"), ("wild", "appearance")),
             (("gen", "quality"), ("wild", "appearance")), (("gen", None), ("wild", None))):
    print(f"  {a} vs {b}: OR {spearman(col(*a, 'OR'), col(*b, 'OR')):+.2f}  DO {spearman(col(*a, 'DO'), col(*b, 'DO')):+.2f}")
