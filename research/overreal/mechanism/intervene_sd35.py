"""Attention intervention on SD3.5-Large (see intervene_common.py).
Usage: CUDA_VISIBLE_DEVICES=0 python intervene_sd35.py --shard 0/6
"""
import math
import os
import sys

os.environ.setdefault("HF_HOME", "/data/users/jiahao_huang/hf")
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attn_sd35 import Model as Base, N_CLIP, N_T5, SIZE
from intervene_common import Bias, argparser, run


class Model(Base):
    def make_mask(self, pipe, text, idx, logk):
        """[B=2,1,N,N] additive mask: image-query rows, chosen text columns,
        conditional batch item only (CFG batch = [uncond, cond])."""
        n_img, n_txt = self.n_img, N_CLIP + N_T5
        N = n_img + n_txt
        m = torch.zeros(2, 1, N, N, dtype=torch.bfloat16, device="cuda")
        cols = torch.tensor([n_img + i for i in idx], device="cuda")
        m[1, 0, :(N if Bias.rows == "all" else n_img), cols] = logk
        return m

    def install(self, pipe):
        from diffusers.models.attention_processor import JointAttnProcessor2_0

        class P(JointAttnProcessor2_0):
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
                mask = None
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
                    if Bias.mask is not None and Bias.mask.shape[0] == batch_size and Bias.mask.shape[-1] == query.shape[2]:
                        mask = Bias.mask
                hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=mask,
                                                               dropout_p=0.0, is_causal=False)
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

        for blk in pipe.transformer.transformer_blocks:
            blk.attn.set_processor(P())


if __name__ == "__main__":
    run(Model(), argparser().parse_args())
