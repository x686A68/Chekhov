"""Attention intervention (R16): shared pieces.

For each item of intervene_items.json (S/P pairs with an SD3.5-L baseline
label in the Table 1 sample) and each seed, generate the image under a set
of conditions that add a logit bias log(k) to the image-to-text attention
scores of chosen text tokens, in every block and step, conditional branch
only. Adding log(k) multiplies those tokens' attention probability by k
before renormalization.

Conditions (side, span kind, factor):
  cue_x2/4/8   S  cue words (the words the control replaces)       x k
  tgt_d2/4/8   S  target words                                     / k
  rand_x8      S  one content word outside target and cue          x 8
  P_rep_x8     P  the replacing words of the plain-mention control x 8
  P_tgt_d8     P  target words                                     / 8
Output: intervene/<model>/<cond>/<item>__s<seed>.jpg + manifest_<shard>.jsonl
"""
import json
import math
import os
import random
import re
import time

from attn_common import diff_spans

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(ROOT, "..", "..", ".."))
PAIRS = os.path.join(ROOT, "pairs.jsonl")
ITEMS = os.path.join(ROOT, "intervene_items.json")
WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")
SEEDS = [0, 1]

CONDS = [
    ("cue_x2", "S", "cue", math.log(2)), ("cue_x4", "S", "cue", math.log(4)), ("cue_x8", "S", "cue", math.log(8)),
    ("tgt_d2", "S", "target", -math.log(2)), ("tgt_d4", "S", "target", -math.log(4)), ("tgt_d8", "S", "target", -math.log(8)),
    ("rand_x8", "S", "random", math.log(8)),
    ("P_rep_x8", "P", "cue", math.log(8)),
    ("P_tgt_d8", "P", "target", -math.log(8)),
    ("cue_x8_all", "S", "cue", math.log(8)),   # bias on every query row (text rows too), see Bias.rows
    ("func_x8", "S", "func", math.log(8)),     # one function word x 8: a neutral magnitude control
]
FUNC = {"a", "an", "the", "of", "in", "on", "at", "to", "and", "with", "for", "from", "by", "her", "his", "its"}


class Bias:
    """Holder the attention processors read: additive mask [B,1,N,N] or None.
    rows = "img" (image queries only) or "all" (text queries too)."""
    mask = None
    rows = "img"


def spans_for(r, side, kind):
    text = r["s_text"] if side == "S" else r["p_text"]
    if kind == "target":
        return r["s_spans"] if side == "S" else r["p_spans"]
    if kind == "cue":
        # the words the control replaces (S) / the words that replace them (P): word-level diff
        return diff_spans(r["s_text"], r["p_text"]) if side == "S" else diff_spans(r["p_text"], r["s_text"])
    taken = (r["s_spans"] or []) + diff_spans(r["s_text"], r["p_text"])
    if kind == "func":
        # one function word outside target and cue spans, deterministic per item
        cands = [(m.start(), m.end()) for m in WORD.finditer(text)
                 if m.group(0).lower() in FUNC and not any(m.start() < e and m.end() > s for s, e in taken)]
        return [random.Random(r["item_id"] + "f").choice(cands)] if cands else []
    # random: a content word (>= 4 letters) outside target and cue spans, deterministic per item
    cands = [(m.start(), m.end()) for m in WORD.finditer(text)
             if m.end() - m.start() >= 4 and not any(m.start() < e and m.end() > s for s, e in taken)]
    if not cands:
        return []
    return [random.Random(r["item_id"]).choice(cands)]


def load_jobs(shard, seeds=SEEDS, items_file=None, conds=None):
    """Filter by condition first, then shard, so every shard holds every
    condition and shards launched with different filters stay consistent."""
    items = set(json.load(open(items_file or ITEMS)))
    pairs = [json.loads(l) for l in open(PAIRS)]
    pairs = [r for r in pairs if r["item_id"] in items and not r["exclude"]]
    keep = set(conds.split(",")) if conds else None
    jobs = [(r, seed, c) for r in pairs for seed in seeds for c in CONDS if keep is None or c[0] in keep]
    i, n = map(int, shard.split("/"))
    return jobs[i::n]


def load_manifest(model_key):
    path = os.path.join(REPO, "data", "generation", "manifests", f"{model_key}_raw.jsonl")
    m = {}
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            m[(r["item_id"], r["seed"])] = r
    return m


def run(model, args):
    import torch
    out_root = os.path.join(ROOT, "intervene", model.key)
    os.makedirs(out_root, exist_ok=True)
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else SEEDS
    jobs = load_jobs(args.shard, seeds, args.items or None, args.conds or None)
    if args.limit:
        jobs = jobs[:args.limit]
    pending = [j for j in jobs if not os.path.exists(
        os.path.join(out_root, j[2][0], f"{j[0]['item_id'].replace('/', '_')}__s{j[1]}.jpg"))]
    print(f"{len(pending)} of {len(jobs)} jobs to do", flush=True)
    if not pending:
        return
    pipe = model.load()
    model.install(pipe)
    manifest = load_manifest(model.key)
    mpath = os.path.join(out_root, f"manifest_{args.shard.replace('/', 'of')}.jsonl")
    t0 = time.time()
    for n, (r, seed, (cname, side, kind, logk)) in enumerate(pending, 1):
        text = r["s_text"] if side == "S" else r["p_text"]
        spans = spans_for(r, side, kind)
        idx, _ = model.tokens(pipe, text, spans)
        mrow = manifest.get((r["item_id"], seed))
        steps = mrow["steps"] if mrow else model.steps_default
        cfg = mrow["cfg"] if mrow else model.cfg_default
        Bias.rows = "all" if cname.endswith("_all") else "img"
        Bias.mask = model.make_mask(pipe, text, idx, logk) if idx else None
        img = model.generate(pipe, text, steps, cfg, seed)
        Bias.mask = None
        os.makedirs(os.path.join(out_root, cname), exist_ok=True)
        fn = os.path.join(out_root, cname, f"{r['item_id'].replace('/', '_')}__s{seed}.jpg")
        img.convert("RGB").save(fn, quality=92)
        with open(mpath, "a") as f:
            f.write(json.dumps(dict(item_id=r["item_id"], family=r["family"], cond=cname, side=side,
                                    kind=kind, logk=logk, seed=seed, steps=steps, cfg=cfg, prompt=text,
                                    spans=spans, n_tokens=len(idx), target=r["target"], file=fn)) + "\n")
        if n % 20 == 0 or n == len(pending):
            print(f"[{n}/{len(pending)}] {time.time() - t0:.0f}s", flush=True)


def argparser():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--conds", default="")
    ap.add_argument("--seeds", default="")
    ap.add_argument("--items", default="")
    return ap
