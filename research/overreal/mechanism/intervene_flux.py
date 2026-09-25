"""Attention intervention on FLUX.1-dev (see intervene_common.py).
Joint sequence = [512 T5 text tokens, 4096 image tokens]; no CFG batch.
Usage: CUDA_VISIBLE_DEVICES=0 python intervene_flux.py --shard 0/3 --seeds 0 --conds cue_x8,tgt_d8,rand_x8,P_rep_x8
"""
import os
import sys

os.environ.setdefault("HF_HOME", "/data/users/jiahao_huang/hf")
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attn_flux import MODEL, N_TXT, SIZE, STEPS_DEFAULT, CFG_DEFAULT, token_indices
from intervene_common import Bias, argparser, run


class Model:
    key = "flux"
    steps_default, cfg_default = STEPS_DEFAULT, CFG_DEFAULT
    n_img = (SIZE // 16) ** 2

    def load(self):
        from diffusers import FluxPipeline
        return FluxPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda")

    def tokens(self, pipe, text, spans):
        return token_indices(pipe.tokenizer_2, text, spans)

    def make_mask(self, pipe, text, idx, logk):
        N = N_TXT + self.n_img
        m = torch.zeros(1, 1, N, N, dtype=torch.bfloat16, device="cuda")
        cols = torch.tensor(idx, device="cuda")
        m[0, 0, (0 if Bias.rows == "all" else N_TXT):, cols] = logk   # image-query rows (or all rows), chosen text columns
        return m

    def install(self, pipe):
        from diffusers.models.transformers.transformer_flux import (FluxAttnProcessor, _get_qkv_projections,
                                                                      apply_rotary_emb)

        class P(FluxAttnProcessor):
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
                mask = Bias.mask if (Bias.mask is not None and Bias.mask.shape[-1] == query.shape[1]) else None
                # [B, N, H, D] -> [B, H, N, D]
                out = F.scaled_dot_product_attention(query.transpose(1, 2), key.transpose(1, 2),
                                                     value.transpose(1, 2), attn_mask=mask)
                hidden_states = out.transpose(1, 2).flatten(2, 3).to(query.dtype)
                if encoder_hidden_states is not None:
                    encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                        [encoder_hidden_states.shape[1],
                         hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1)
                    hidden_states = attn.to_out[0](hidden_states.contiguous())
                    hidden_states = attn.to_out[1](hidden_states)
                    encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
                    return hidden_states, encoder_hidden_states
                return hidden_states

        for blk in list(pipe.transformer.transformer_blocks) + list(pipe.transformer.single_transformer_blocks):
            blk.attn.set_processor(P())

    def generate(self, pipe, text, steps, cfg, seed):
        gen = torch.Generator("cuda").manual_seed(seed)
        return pipe(prompt=text, num_inference_steps=steps, guidance_scale=cfg, width=SIZE,
                    height=SIZE, generator=gen, max_sequence_length=N_TXT).images[0]


if __name__ == "__main__":
    run(Model(), argparser().parse_args())
