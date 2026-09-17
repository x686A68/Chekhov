"""Shared pieces of the attention recorders (attn_flux.py, attn_sd35.py,
attn_qwen.py): the recorder, token-span utilities, the manifest, and the
per-image driver loop. Each model script supplies a `Model` object with
  load()                      -> pipeline
  install(pipe, recorder)     -> installs recording processors
  tokens(pipe, text, spans)   -> (token indices, n_real) in the joint text seq
  n_blocks, n_txt(pipe, text), n_img
  generate(pipe, text, steps, cfg, seed, on_step) -> PIL image
"""
import difflib
import json
import os
import re
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(ROOT, "..", "..", ".."))
PAIRS = os.path.join(ROOT, "pairs.jsonl")
N_BINS = 5
SEEDS = [0, 1]
WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")


class Recorder:
    """Accumulates, per block and step, the mean over image queries and heads
    of the attention probability to each text token, and a spatial map of the
    attention to the target tokens in N_BINS step bins."""

    def __init__(self, n_blocks, n_steps, n_txt, n_img, target_idx):
        self.mass = torch.zeros(n_blocks, n_steps, n_txt, dtype=torch.float32, device="cuda")
        self.map = torch.zeros(N_BINS, n_img, dtype=torch.float32, device="cuda")
        self.map_count = torch.zeros(N_BINS, device="cuda")
        self.n_txt, self.n_steps, self.n_img = n_txt, n_steps, n_img
        self.target_idx = torch.tensor(target_idx, device="cuda")
        self.step = -1   # advanced when block 0 is recorded (once per step)

    @torch.no_grad()
    def record(self, block, q, k, txt, img):
        """q, k: [H, N, D] for the conditional sample; txt, img: slices of the
        joint sequence holding the text and image tokens."""
        if block == 0:
            self.step += 1
        H, N, D = q.shape
        scale = D ** -0.5
        # all heads at once: [H, n_img, N] scores in bf16, softmax pieces in fp32
        s = torch.matmul(q[:, img], k.transpose(1, 2)).float() * scale
        lse = torch.logsumexp(s, dim=-1, keepdim=True)
        acc = torch.exp(s[:, :, txt] - lse).mean(0)          # [n_img, n_txt]
        del s, lse
        step = min(self.step, self.n_steps - 1)
        self.mass[block, step] += acc.mean(0)
        b = min(step * N_BINS // self.n_steps, N_BINS - 1)
        self.map[b] += acc[:, self.target_idx].sum(1)
        self.map_count[b] += 1


def span_token_indices(offsets, spans, prefix=0):
    idx, n_real = [], 0
    for i, (a, b) in enumerate(offsets):
        if b <= a:
            continue
        n_real += 1
        a -= prefix
        b -= prefix
        if any(a < e and b > s for s, e in spans):
            idx.append(i)
    return idx, n_real


def diff_spans(text_a, text_b):
    """Character spans of the words of text_a absent from text_b."""
    wa = [(m.group(0).lower(), m.start(), m.end()) for m in WORD.finditer(text_a)]
    wb = [m.group(0).lower() for m in WORD.finditer(text_b)]
    sm = difflib.SequenceMatcher(a=[w for w, _, _ in wa], b=wb, autojunk=False)
    spans = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "delete"):
            for w, s, e in wa[i1:i2]:
                spans.append([s, e])
    return spans


def expanded_jobs(cond, seeds=SEEDS):
    """All expanded prompts of one rewriter in which the target string of the
    item survives verbatim, both seeds. Items excluded from pairs.jsonl are
    excluded here too."""
    pairs = {r["item_id"]: r for r in (json.loads(l) for l in open(PAIRS)) if not r["exclude"]}
    rows = [json.loads(l) for l in open(os.path.join(REPO, "data", "generation", "expanded_prompts.jsonl"))
            if l.strip()]
    jobs = []
    for rr in rows:
        if rr["expander"] != cond or rr["item_id"] not in pairs:
            continue
        pr = pairs[rr["item_id"]]
        t = pr["target"].strip().strip('"').strip("'").rstrip(".").strip()
        m = re.search(re.escape(t), rr["text"], flags=re.IGNORECASE)
        if not m:
            continue
        r = {"item_id": rr["item_id"], "family": rr["family"], "target": pr["target"],
             "x_text": rr["text"], "x_spans": [[m.start(), m.end()]]}
        jobs += [(r, "X", sd) for sd in seeds]
    return jobs


def load_manifest(model_key, cond="raw"):
    path = os.path.join(REPO, "data", "generation", "manifests", f"{model_key}_{cond}.jsonl")
    m = {}
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            m[(r["item_id"], r["seed"])] = r
    return m


def run(model, args):
    k, n = map(int, args.shard.split("/"))
    pairs = [r for r in (json.loads(l) for l in open(PAIRS)) if not r["exclude"]]
    if args.items:
        want = set(args.items.split(","))
        pairs = [r for r in pairs if r["item_id"] in want]
    pairs = pairs[k::n]
    if args.limit:
        pairs = pairs[:args.limit]
    manifest = load_manifest(model.key, args.cond)
    out_dir = os.path.join(ROOT, "attn", model.key + ("" if args.cond == "raw" else "_" + args.cond))
    os.makedirs(out_dir, exist_ok=True)

    pipe = model.load()
    if args.cond == "raw":
        jobs = [(r, side, seed) for r in pairs for side in args.sides for seed in SEEDS]
    else:
        jobs = expanded_jobs(args.cond)[k::n]
        if args.limit:
            jobs = jobs[:args.limit]
    print(f"{model.key} shard {k}/{n}: {len(jobs)} images", flush=True)
    for r, side, seed in jobs:
        name = f"{r['item_id'].replace('/', '_')}__{side}_s{seed}"
        out_npz = os.path.join(out_dir, name + ".npz")
        if os.path.exists(out_npz):
            continue
        if side == "X":                       # expanded prompt: no control, no cue
            text, spans, other = r["x_text"], r["x_spans"], None
        else:
            text = r["s_text"] if side == "S" else r["p_text"]
            spans = r["s_spans"] if side == "S" else r["p_spans"]
            other = r["p_text"] if side == "S" else r["s_text"]
        target_idx, n_real = model.tokens(pipe, text, spans)
        cue_idx = model.tokens(pipe, text, diff_spans(text, other))[0] if other else []
        real_idx = model.real_indices(pipe, text) if hasattr(model, "real_indices") else np.arange(n_real)
        mrow = manifest.get((r["item_id"], seed))
        steps = mrow["steps"] if mrow else model.steps_default
        cfg = mrow["cfg"] if mrow else model.cfg_default

        rec = Recorder(model.n_blocks, steps, model.n_txt(pipe, text), model.n_img, target_idx)
        model.install(pipe, rec)

        t0 = time.time()
        img = model.generate(pipe, text, steps, cfg, seed)
        sec = time.time() - t0

        img.save(os.path.join(out_dir, name + ".png"))
        np.savez_compressed(
            out_npz,
            mass=rec.mass.half().cpu().numpy(),
            map=(rec.map / rec.map_count.clamp(min=1)[:, None]).half().cpu().numpy(),
            target_idx=np.array(target_idx), cue_idx=np.array(cue_idx), n_real=n_real,
            real_idx=np.array(real_idx),
            item_id=r["item_id"], family=r["family"], side=side, seed=seed, steps=steps,
            cfg=cfg, prompt=text, target=r["target"])
        msg = f"{name} {sec:.0f}s target_tok={target_idx} cue_tok={cue_idx}"
        if args.check and side in "SX" and mrow:
            from PIL import Image
            ref = np.asarray(Image.open(os.path.join(REPO, "data", mrow["file"])).convert("RGB")).astype(np.float32)
            cur = np.asarray(img.convert("RGB")).astype(np.float32)
            msg += f" | mean|diff| vs stored = {np.abs(ref - cur).mean():.2f}/255"
        print(msg, flush=True)


def argparser():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sides", default="SP")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--items", default="", help="comma-separated item ids to run (default: all)")
    ap.add_argument("--cond", default="raw", help="raw, or an expander (qwen, ideogram): expanded prompts")
    return ap
