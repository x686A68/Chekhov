"""Extract text-encoder features for the S/P pairs, exactly as the T2I
pipelines compute them (same weights, tokenizer, padding and template).

For every item and both sides (S, P) and every layer:
  span  : mean of the hidden states over the target span tokens
  ctrl  : same for the control word span (NaN when the item has none)
  mean  : mean over the prompt's real tokens (no pad, no special tokens)
CLIP encoders additionally store the pooled vectors the pipelines use:
  pooled: pooler_output (FLUX uses this from CLIP-L)
  proj  : text_embeds, the projected pooled vector (SD3 uses this)

Encoders (loaded from the pipeline repos in HF_HOME):
  clip_l  SD3.5-L text_encoder    CLIP ViT-L/14, 77 tokens, 12 layers
  clip_g  SD3.5-L text_encoder_2  CLIP ViT-bigG/14, 77 tokens, 32 layers
  t5      SD3.5-L text_encoder_3  T5-XXL encoder, 256 tokens, no attention
                                  mask (as diffusers does), 24 layers
  qwen    Qwen-Image text_encoder Qwen2.5-VL-7B, describe-the-image template,
                                  first 34 template tokens dropped, 28 layers

Output: features/<encoder>.npz with float16 arrays
  span_S, span_P, ctrl_S, ctrl_P, mean_S, mean_P : [n, n_layers+1, dim]
  pooled_S, pooled_P, proj_S, proj_P (CLIP only) : [n, dim]
  item_ids, families : [n]
Layer index 0 is the embedding output; the last index is the final layer.

Usage: CUDA_VISIBLE_DEVICES=0 python extract_features.py --encoder t5
"""
import argparse
import json
import os

os.environ.setdefault("HF_HOME", "/data/users/jiahao_huang/hf")

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
PAIRS = os.path.join(ROOT, "pairs.jsonl")
OUT_DIR = os.path.join(ROOT, "features")

SD35 = "stabilityai/stable-diffusion-3.5-large"
QWEN_IMAGE = "Qwen/Qwen-Image"
QWEN_TEMPLATE = ("<|im_start|>system\nDescribe the image by detailing the color, "
                 "shape, size, texture, quantity, text, spatial relationships of "
                 "the objects and background:<|im_end|>\n<|im_start|>user\n{}"
                 "<|im_end|>\n<|im_start|>assistant\n")
QWEN_DROP = 34


def load_pairs():
    rows = [json.loads(l) for l in open(PAIRS) if l.strip()]
    return [r for r in rows if not r["exclude"]]


def token_mask(offsets, spans, prefix=0):
    """Boolean mask over tokens whose character range overlaps any span."""
    m = np.zeros(len(offsets), dtype=bool)
    for i, (a, b) in enumerate(offsets):
        if b <= a:
            continue  # special or pad token
        a -= prefix
        b -= prefix
        for s, e in spans:
            if a < e and b > s:
                m[i] = True
    return m


class Encoder:
    n_layers = None
    dim = None

    def encode(self, texts):
        """Return hidden [B, L+1, T, D] (float32 cpu), real-token mask [B, T],
        offsets list of lists, prefix length, and optional dict of pooled."""
        raise NotImplementedError


class ClipEncoder(Encoder):
    def __init__(self, which, device):
        from transformers import CLIPTokenizer, CLIPTextModelWithProjection
        sub = {"clip_l": ("tokenizer", "text_encoder"),
               "clip_g": ("tokenizer_2", "text_encoder_2")}[which]
        self.tok = CLIPTokenizer.from_pretrained(SD35, subfolder=sub[0])
        self.model = CLIPTextModelWithProjection.from_pretrained(
            SD35, subfolder=sub[1], torch_dtype=torch.bfloat16).to(device).eval()
        self.device = device
        self.max_len = self.tok.model_max_length  # 77

    @torch.no_grad()
    def encode(self, texts):
        enc = self.tok(texts, padding="max_length", max_length=self.max_len,
                       truncation=True, return_tensors="pt",
                       return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping").tolist()
        out = self.model(enc["input_ids"].to(self.device), output_hidden_states=True)
        hs = torch.stack(out.hidden_states, dim=1).float().cpu()  # [B, L+1, T, D]
        real = (enc["attention_mask"] == 1)
        # drop BOS/EOS from the mean: they have zero-length offsets
        real = real & torch.tensor([[b > a for a, b in o] for o in offsets])
        pooled = {"pooled": out.text_embeds.new_tensor(0)}  # placeholder replaced below
        pooled = {"proj": out.text_embeds.float().cpu()}
        # pooler_output (pre-projection) as used by FLUX
        last = out.hidden_states[-1]
        last = self.model.text_model.final_layer_norm(last)
        eos = enc["input_ids"].to(self.device).argmax(dim=-1)  # EOS has the max id in CLIP
        pooled["pooled"] = last[torch.arange(last.shape[0]), eos].float().cpu()
        return hs, real, offsets, 0, pooled


class T5Encoder(Encoder):
    def __init__(self, device):
        from transformers import T5TokenizerFast, T5EncoderModel
        self.tok = T5TokenizerFast.from_pretrained(SD35, subfolder="tokenizer_3")
        self.model = T5EncoderModel.from_pretrained(
            SD35, subfolder="text_encoder_3", torch_dtype=torch.bfloat16).to(device).eval()
        self.device = device
        self.max_len = 256  # SD3 default max_sequence_length

    @torch.no_grad()
    def encode(self, texts):
        enc = self.tok(texts, padding="max_length", max_length=self.max_len,
                       truncation=True, add_special_tokens=True, return_tensors="pt",
                       return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping").tolist()
        # diffusers passes input_ids only: padded positions are attended
        out = self.model(enc["input_ids"].to(self.device), output_hidden_states=True)
        hs = torch.stack(out.hidden_states, dim=1).float().cpu()
        real = (enc["attention_mask"] == 1) & torch.tensor(
            [[b > a for a, b in o] for o in offsets])
        return hs, real, offsets, 0, {}


class QwenEncoder(Encoder):
    def __init__(self, device):
        from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration
        self.tok = AutoTokenizer.from_pretrained(QWEN_IMAGE, subfolder="tokenizer")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            QWEN_IMAGE, subfolder="text_encoder", torch_dtype=torch.bfloat16).to(device).eval()
        self.device = device
        self.max_len = 1024 + QWEN_DROP
        self.prefix = QWEN_TEMPLATE.index("{}")

    @torch.no_grad()
    def encode(self, texts):
        txt = [QWEN_TEMPLATE.format(t) for t in texts]
        enc = self.tok(txt, max_length=self.max_len, padding=True, truncation=True,
                       return_tensors="pt", return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping").tolist()
        out = self.model(input_ids=enc["input_ids"].to(self.device),
                         attention_mask=enc["attention_mask"].to(self.device),
                         output_hidden_states=True)
        hs = torch.stack(out.hidden_states, dim=1).float().cpu()
        real = (enc["attention_mask"] == 1)
        # real prompt tokens: after the dropped template prefix, before the
        # closing template; offsets inside the user prompt span
        end = self.prefix + 0
        real = real.clone()
        for i, o in enumerate(offsets):
            plen = len(texts[i])
            for j, (a, b) in enumerate(o):
                if not (b > a and a >= self.prefix and b <= self.prefix + plen):
                    real[i, j] = False
        return hs, real, offsets, self.prefix, {}


def build(which, device):
    if which in ("clip_l", "clip_g"):
        return ClipEncoder(which, device)
    if which == "t5":
        return T5Encoder(device)
    if which == "qwen":
        return QwenEncoder(device)
    raise ValueError(which)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, choices=["clip_l", "clip_g", "t5", "qwen"])
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="smoke test on N items")
    args = ap.parse_args()

    rows = load_pairs()
    if args.limit:
        rows = rows[:args.limit]
    bs = args.batch or {"clip_l": 64, "clip_g": 64, "t5": 16, "qwen": 8}[args.encoder]
    enc = build(args.encoder, args.device)

    feats = {k: [] for k in ["span_S", "span_P", "ctrl_S", "ctrl_P", "mean_S", "mean_P",
                             "pooled_S", "pooled_P", "proj_S", "proj_P"]}
    shown = 0
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        for side in ("S", "P"):
            texts = [r[f"{side.lower()}_text"] for r in chunk]
            hs, real, offsets, prefix, pooled = enc.encode(texts)
            for j, r in enumerate(chunk):
                tspan = r[f"{side.lower()}_spans"]
                cspan = r["c_spans_s" if side == "S" else "c_spans_p"]
                tm = token_mask(offsets[j], tspan, prefix)
                if not tm.any():
                    raise SystemExit(f"no tokens for target span {r['item_id']} {side}: "
                                     f"{tspan} in {texts[j]!r}")
                h = hs[j]  # [L+1, T, D]
                feats[f"span_{side}"].append(h[:, tm].mean(1).half().numpy())
                if cspan:
                    cm = token_mask(offsets[j], cspan, prefix)
                    feats[f"ctrl_{side}"].append(h[:, cm].mean(1).half().numpy())
                else:
                    feats[f"ctrl_{side}"].append(np.full(h.shape[:1] + h.shape[2:], np.nan, np.float16))
                rm = real[j].numpy()
                feats[f"mean_{side}"].append(h[:, rm].mean(1).half().numpy())
                for k in ("pooled", "proj"):
                    if k in pooled:
                        feats[f"{k}_{side}"].append(pooled[k][j].half().numpy())
                if shown < 3 and side == "S":
                    ids = enc.tok(texts[j], add_special_tokens=True)["input_ids"] \
                        if args.encoder != "qwen" else \
                        enc.tok(QWEN_TEMPLATE.format(texts[j]))["input_ids"]
                    sel = [enc.tok.decode([ids[t]]) for t in np.where(tm)[0] if t < len(ids)]
                    print(f"[check] {r['item_id']} target={r['target']!r} tokens={sel}")
                    shown += 1
        print(f"{min(i + bs, len(rows))}/{len(rows)}", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    out = {k: np.stack(v) for k, v in feats.items() if v}
    out["item_ids"] = np.array([r["item_id"] for r in rows])
    out["families"] = np.array([r["family"] for r in rows])
    path = os.path.join(OUT_DIR, f"{args.encoder}{'_smoke' if args.limit else ''}.npz")
    np.savez_compressed(path, **out)
    print("saved", path, {k: v.shape for k, v in out.items() if hasattr(v, "shape")})


if __name__ == "__main__":
    main()
