"""Attention recorder for SD3.5-Large (see attn_common.py).

Joint sequence in SD3: [image (4096), text (77 CLIP + 256 T5 = 333)].
Batch under CFG is [uncond, cond]; the conditional sample is index 1.
Target and cue tokens are located in both the CLIP part (positions 0..76)
and the T5 part (77..332).

Usage: CUDA_VISIBLE_DEVICES=0 python attn_sd35.py --shard 0/6
"""
import os

os.environ.setdefault("HF_HOME", "/data/users/jiahao_huang/hf")

import torch
import torch.nn.functional as F

from attn_common import argparser, run, span_token_indices

MODEL = "stabilityai/stable-diffusion-3.5-large"
SIZE = 1024
N_CLIP, N_T5 = 77, 256


class Model:
    key = "sd35l"
    steps_default, cfg_default = 40, 4.5
    n_blocks = 38
    n_img = (SIZE // 16) ** 2

    def load(self):
        from diffusers import StableDiffusion3Pipeline
        return StableDiffusion3Pipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda")

    def n_txt(self, pipe, text):
        return N_CLIP + N_T5

    def tokens(self, pipe, text, spans):
        enc = pipe.tokenizer(text, padding="max_length", max_length=N_CLIP, truncation=True,
                             return_offsets_mapping=True)
        ci, n1 = span_token_indices(enc["offset_mapping"], spans)
        enc = pipe.tokenizer_3(text, padding="max_length", max_length=N_T5, truncation=True,
                               add_special_tokens=True, return_offsets_mapping=True)
        ti, n2 = span_token_indices(enc["offset_mapping"], spans)
        return ci + [N_CLIP + i for i in ti], n1 + n2

    def real_indices(self, pipe, text):
        """Positions of the prompt's own tokens (no BOS/EOS/pad) in the
        [77 CLIP + 256 T5] text sequence."""
        idx = []
        enc = pipe.tokenizer(text, padding="max_length", max_length=N_CLIP, truncation=True,
                             return_offsets_mapping=True)
        idx += [i for i, (a, b) in enumerate(enc["offset_mapping"]) if b > a]
        enc = pipe.tokenizer_3(text, padding="max_length", max_length=N_T5, truncation=True,
                               add_special_tokens=True, return_offsets_mapping=True)
        idx += [N_CLIP + i for i, (a, b) in enumerate(enc["offset_mapping"]) if b > a]
        return idx

    def install(self, pipe, rec):
        from diffusers.models.attention_processor import JointAttnProcessor2_0
        n_img = self.n_img

        class P(JointAttnProcessor2_0):
            def __init__(self, block):
                super().__init__()
                self.block = block

            def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None,
                         *args, **kwargs):
                residual = hidden_states
                batch_size = hidden_states.shape[0]
                query = attn.to_q(hidden_states)
                key = attn.to_k(hidden_states)
                value = attn.to_v(hidden_states)
                inner_dim = key.shape[-1]
                head_dim = inner_dim // attn.heads
                query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                if attn.norm_q is not None:
                    query = attn.norm_q(query)
                if attn.norm_k is not None:
                    key = attn.norm_k(key)
                if encoder_hidden_states is not None:
                    eq = attn.add_q_proj(encoder_hidden_states)
                    ek = attn.add_k_proj(encoder_hidden_states)
                    ev = attn.add_v_proj(encoder_hidden_states)
                    eq = eq.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                    ek = ek.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                    ev = ev.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                    if attn.norm_added_q is not None:
                        eq = attn.norm_added_q(eq)
                    if attn.norm_added_k is not None:
                        ek = attn.norm_added_k(ek)
                    query = torch.cat([query, eq], dim=2)
                    key = torch.cat([key, ek], dim=2)
                    value = torch.cat([value, ev], dim=2)
                    cond = 1 if batch_size == 2 else 0
                    n_seq = query.shape[2]
                    rec.record(self.block, query[cond], key[cond],
                               txt=slice(n_img, n_seq), img=slice(0, n_img))
                hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
                hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
                hidden_states = hidden_states.to(query.dtype)
                if encoder_hidden_states is not None:
                    hidden_states, encoder_hidden_states = (
                        hidden_states[:, : residual.shape[1]], hidden_states[:, residual.shape[1]:])
                    if not attn.context_pre_only:
                        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)
                if encoder_hidden_states is not None:
                    return hidden_states, encoder_hidden_states
                return hidden_states

        for i, blk in enumerate(pipe.transformer.transformer_blocks):
            blk.attn.set_processor(P(i))

    def generate(self, pipe, text, steps, cfg, seed):
        gen = torch.Generator("cuda").manual_seed(seed)
        return pipe(prompt=text, num_inference_steps=steps, guidance_scale=cfg, width=SIZE,
                    height=SIZE, generator=gen, max_sequence_length=N_T5).images[0]


if __name__ == "__main__":
    run(Model(), argparser().parse_args())
