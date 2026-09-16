"""Regenerate OverReal-Gen images with FLUX.1-dev under the benchmark seeds
and record, at every transformer block and denoising step, how much
attention the image tokens pay to each text token.

For each (item, side, seed):
  mass  [n_blocks, n_steps, 512]  mean over image queries and heads of the
                                  attention probability to each text token
  map   [5, 4096]                 attention from each image token to the
                                  target tokens, averaged over blocks and
                                  heads, in five step bins (early -> late)
  meta                            target token indices, cue token indices,
                                  real token count, prompt, seed, steps
The image itself is saved next to the record so labels can be checked.

The attention actually used for generation is untouched (SDPA as usual); the
probabilities are recomputed on the side from the same q and k, so the image
is the same one the benchmark scored when seed, steps and cfg match the
manifest.

Sides: S = original prompt, P = plain-mention control (pairs.jsonl).
Cue tokens = tokens of the words that occur in S but not in P (the phrase the
rewrite replaced); for the P side, the words in P but not in S.

Usage: CUDA_VISIBLE_DEVICES=0 python attn_flux.py --shard 0/4 [--limit 2 --check]
"""
import argparse
import difflib
import json
import os
import re
import time

os.environ.setdefault("HF_HOME", "/data/users/jiahao_huang/hf")

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(ROOT, "..", "..", ".."))
PAIRS = os.path.join(ROOT, "pairs.jsonl")
MANIFEST = os.path.join(REPO, "data", "generation", "manifests", "flux_raw.jsonl")
IMG_DIR = os.path.join(REPO, "data", "generation", "images", "flux", "raw")
OUT = os.path.join(ROOT, "attn", "flux")

MODEL = "black-forest-labs/FLUX.1-dev"
SEEDS = [0, 1]
STEPS_DEFAULT, CFG_DEFAULT, SIZE = 50, 3.5, 1024
N_TXT = 512
N_BINS = 5

WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")


# ----------------------------------------------------------------------------
# recording attention processor
# ----------------------------------------------------------------------------
class Recorder:
    def __init__(self, n_blocks, n_steps, n_txt, n_img, target_idx):
        self.mass = torch.zeros(n_blocks, n_steps, n_txt, dtype=torch.float32, device="cuda")
        self.map = torch.zeros(N_BINS, n_img, dtype=torch.float32, device="cuda")
        self.map_count = torch.zeros(N_BINS, device="cuda")
        self.n_txt, self.n_steps = n_txt, n_steps
        self.target_idx = torch.tensor(target_idx, device="cuda")
        self.step = -1

    def record(self, block, query, key):
        """query, key: [B, N, H, D] after rotary. Batch 1 (FLUX has no CFG batch)."""
        q = query[0].transpose(0, 1)  # [H, N, D]
        k = key[0].transpose(0, 1)
        H, N, D = q.shape
        n_txt = self.n_txt
        scale = D ** -0.5
        s = torch.matmul(q[:, n_txt:], k.transpose(1, 2)).float() * scale   # [H, n_img, N]
        lse = torch.logsumexp(s, dim=-1, keepdim=True)
        acc = torch.exp(s[:, :, :n_txt] - lse).mean(0)                      # [n_img, n_txt]
        del s, lse
        self.mass[block, self.step] += acc.mean(0)
        b = min(self.step * N_BINS // self.n_steps, N_BINS - 1)
        self.map[b] += acc[:, self.target_idx].sum(1)
        self.map_count[b] += 1


def make_processor(base_cls, recorder, block_index):
    from diffusers.models.transformers.transformer_flux import (_get_qkv_projections,
                                                                  apply_rotary_emb)
    from diffusers.models.attention_dispatch import dispatch_attention_fn

    class RecordingProcessor(base_cls):
        def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                     attention_mask=None, image_rotary_emb=None):
            query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
                attn, hidden_states, encoder_hidden_states)
            query = query.unflatten(-1, (-1, attn.head_dim))
            key = key.unflatten(-1, (-1, attn.head_dim))
            value = value.unflatten(-1, (-1, attn.head_dim))
            query = attn.norm_q(query)
            key = attn.norm_k(key)
            if attn.added_kv_proj_dim is not None:
                encoder_query = encoder_query.unflatten(-1, (-1, attn.head_dim))
                encoder_key = encoder_key.unflatten(-1, (-1, attn.head_dim))
                encoder_value = encoder_value.unflatten(-1, (-1, attn.head_dim))
                encoder_query = attn.norm_added_q(encoder_query)
                encoder_key = attn.norm_added_k(encoder_key)
                query = torch.cat([encoder_query, query], dim=1)
                key = torch.cat([encoder_key, key], dim=1)
                value = torch.cat([encoder_value, value], dim=1)
            if image_rotary_emb is not None:
                query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
                key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

            with torch.no_grad():
                recorder.record(block_index, query, key)

            hidden_states = dispatch_attention_fn(
                query, key, value, attn_mask=attention_mask,
                backend=self._attention_backend, parallel_config=self._parallel_config)
            hidden_states = hidden_states.flatten(2, 3).to(query.dtype)
            if encoder_hidden_states is not None:
                encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                    [encoder_hidden_states.shape[1],
                     hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1)
                hidden_states = attn.to_out[0](hidden_states.contiguous())
                hidden_states = attn.to_out[1](hidden_states)
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
                return hidden_states, encoder_hidden_states
            return hidden_states

    return RecordingProcessor()


def install(pipe, recorder):
    from diffusers.models.transformers.transformer_flux import FluxAttnProcessor
    blocks = list(pipe.transformer.transformer_blocks) + list(pipe.transformer.single_transformer_blocks)
    for i, blk in enumerate(blocks):
        blk.attn.set_processor(make_processor(FluxAttnProcessor, recorder, i))
    return len(blocks)


# ----------------------------------------------------------------------------
# tokens
# ----------------------------------------------------------------------------
def token_indices(tok, prompt, spans):
    enc = tok(prompt, padding="max_length", max_length=N_TXT, truncation=True,
              return_offsets_mapping=True)
    idx = []
    n_real = 0
    for i, (a, b) in enumerate(enc["offset_mapping"]):
        if b > a:
            n_real += 1
            if any(a < e and b > s for s, e in spans):
                idx.append(i)
    return idx, n_real


def diff_spans(text_a, text_b):
    """Character spans of the words in text_a that are not in text_b (by
    word-level diff)."""
    wa = [(m.group(0).lower(), m.start(), m.end()) for m in WORD.finditer(text_a)]
    wb = [m.group(0).lower() for m in WORD.finditer(text_b)]
    sm = difflib.SequenceMatcher(a=[w for w, _, _ in wa], b=wb, autojunk=False)
    spans = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "delete"):
            for w, s, e in wa[i1:i2]:
                spans.append([s, e])
    return spans


# ----------------------------------------------------------------------------
def load_manifest():
    m = {}
    if os.path.exists(MANIFEST):
        for line in open(MANIFEST):
            r = json.loads(line)
            m[(r["item_id"], r["seed"])] = r
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sides", default="SP")
    ap.add_argument("--check", action="store_true", help="compare with the stored benchmark image")
    args = ap.parse_args()
    k, n = map(int, args.shard.split("/"))

    from diffusers import FluxPipeline
    pairs = [r for r in (json.loads(l) for l in open(PAIRS)) if not r["exclude"]]
    pairs = pairs[k::n]
    if args.limit:
        pairs = pairs[:args.limit]
    manifest = load_manifest()
    os.makedirs(OUT, exist_ok=True)

    pipe = FluxPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda")
    tok = pipe.tokenizer_2
    n_img = (SIZE // 16) ** 2

    jobs = [(r, side, seed) for r in pairs for side in args.sides for seed in SEEDS]
    print(f"shard {k}/{n}: {len(jobs)} images", flush=True)
    for r, side, seed in jobs:
        name = f"{r['item_id'].replace('/', '_')}__{side}_s{seed}"
        out_npz = os.path.join(OUT, name + ".npz")
        if os.path.exists(out_npz):
            continue
        text = r["s_text"] if side == "S" else r["p_text"]
        spans = r["s_spans"] if side == "S" else r["p_spans"]
        other = r["p_text"] if side == "S" else r["s_text"]
        target_idx, n_real = token_indices(tok, text, spans)
        cue_idx, _ = token_indices(tok, text, diff_spans(text, other))
        mrow = manifest.get((r["item_id"], seed))
        steps = mrow["steps"] if mrow else STEPS_DEFAULT
        cfg = mrow["cfg"] if mrow else CFG_DEFAULT

        rec = Recorder(n_blocks=57, n_steps=steps, n_txt=N_TXT, n_img=n_img, target_idx=target_idx)
        install(pipe, rec)

        def on_step(p, i, t, kw):
            rec.step = i + 1
            return kw
        rec.step = 0
        t0 = time.time()
        gen = torch.Generator("cuda").manual_seed(seed)
        img = pipe(prompt=text, num_inference_steps=steps, guidance_scale=cfg, width=SIZE,
                   height=SIZE, generator=gen, max_sequence_length=N_TXT,
                   callback_on_step_end=on_step).images[0]
        sec = time.time() - t0

        img.save(os.path.join(OUT, name + ".png"))
        np.savez_compressed(
            out_npz,
            mass=rec.mass.half().cpu().numpy(),
            map=(rec.map / rec.map_count.clamp(min=1)[:, None]).half().cpu().numpy(),
            target_idx=np.array(target_idx), cue_idx=np.array(cue_idx), n_real=n_real,
            item_id=r["item_id"], family=r["family"], side=side, seed=seed, steps=steps,
            cfg=cfg, prompt=text, target=r["target"])
        msg = f"{name} {sec:.0f}s target_tok={target_idx} cue_tok={cue_idx}"
        if args.check and side == "S" and mrow:
            from PIL import Image
            ref = np.asarray(Image.open(os.path.join(REPO, "data", mrow["file"])).convert("RGB")).astype(np.float32)
            cur = np.asarray(img.convert("RGB")).astype(np.float32)
            msg += f" | mean|diff| vs stored = {np.abs(ref - cur).mean():.2f}/255"
        print(msg, flush=True)


if __name__ == "__main__":
    main()
