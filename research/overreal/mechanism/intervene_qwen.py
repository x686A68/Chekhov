"""Attention intervention on Qwen-Image (see intervene_common.py).
Joint sequence = [text tokens, 4096 image tokens]; true CFG runs the
conditional and the negative prompt as separate passes, told apart by the
text length. Bias applied to the conditional pass only.
Usage: CUDA_VISIBLE_DEVICES=0 python intervene_qwen.py --shard 0/3 --seeds 0 --conds cue_x8,tgt_d8,rand_x8,P_rep_x8
"""
import os
import sys

os.environ.setdefault("HF_HOME", "/data/users/jiahao_huang/hf")
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attn_qwen import Model as Base, SIZE
from intervene_common import Bias, argparser, run


class Model(Base):
    def make_mask(self, pipe, text, idx, logk):
        n_txt = self.n_txt(pipe, text)
        N = n_txt + self.n_img
        m = torch.zeros(1, 1, N, N, dtype=torch.bfloat16, device="cuda")
        cols = torch.tensor(idx, device="cuda")
        m[0, 0, (0 if Bias.rows == "all" else n_txt):, cols] = logk
        self._n_txt_cond = n_txt
        return m

    def install(self, pipe):
        from diffusers.models.transformers.transformer_qwenimage import QwenDoubleStreamAttnProcessor2_0
        model = self

        class P(QwenDoubleStreamAttnProcessor2_0):
            def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                         encoder_hidden_states_mask=None, attention_mask=None, image_rotary_emb=None):
                from diffusers.models.transformers.transformer_qwenimage import ROPE_PER_DEVICE
                seq_txt = encoder_hidden_states.shape[1]
                img_query = attn.to_q(hidden_states)
                img_key = attn.to_k(hidden_states)
                img_value = attn.to_v(hidden_states)
                txt_query = attn.add_q_proj(encoder_hidden_states)
                txt_key = attn.add_k_proj(encoder_hidden_states)
                txt_value = attn.add_v_proj(encoder_hidden_states)
                head_dim = attn.inner_dim // attn.heads
                img_query = img_query.unflatten(-1, (-1, head_dim))
                img_key = img_key.unflatten(-1, (-1, head_dim))
                img_value = img_value.unflatten(-1, (-1, head_dim))
                txt_query = txt_query.unflatten(-1, (-1, head_dim))
                txt_key = txt_key.unflatten(-1, (-1, head_dim))
                txt_value = txt_value.unflatten(-1, (-1, head_dim))
                if attn.norm_q is not None:
                    img_query = attn.norm_q(img_query)
                if attn.norm_k is not None:
                    img_key = attn.norm_k(img_key)
                if attn.norm_added_q is not None:
                    txt_query = attn.norm_added_q(txt_query)
                if attn.norm_added_k is not None:
                    txt_key = attn.norm_added_k(txt_key)
                if image_rotary_emb is not None:
                    img_freqs, txt_freqs = image_rotary_emb
                    apply_rope = ROPE_PER_DEVICE.get(img_query.device.type, ROPE_PER_DEVICE["cuda"])
                    img_query = apply_rope(img_query, img_freqs)
                    img_key = apply_rope(img_key, img_freqs)
                    txt_query = apply_rope(txt_query, txt_freqs)
                    txt_key = apply_rope(txt_key, txt_freqs)
                q = torch.cat([txt_query, img_query], dim=1).transpose(1, 2)   # [B,H,N,D]
                k = torch.cat([txt_key, img_key], dim=1).transpose(1, 2)
                v = torch.cat([txt_value, img_value], dim=1).transpose(1, 2)
                B, N = q.shape[0], q.shape[2]
                # padding mask (bool, [B, seq_txt]) -> additive float over keys
                mask = None
                if encoder_hidden_states_mask is not None:
                    keep = torch.cat([encoder_hidden_states_mask.bool(),
                                      torch.ones(B, N - seq_txt, dtype=torch.bool, device=q.device)], dim=1)
                    mask = torch.zeros(B, 1, 1, N, dtype=q.dtype, device=q.device)
                    mask = mask.masked_fill(~keep[:, None, None, :], float("-inf"))
                if Bias.mask is not None and seq_txt == getattr(model, "_n_txt_cond", -1) and Bias.mask.shape[-1] == N:
                    mask = Bias.mask if mask is None else mask + Bias.mask
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
                joint = out.transpose(1, 2).flatten(2, 3).to(q.dtype)
                txt_attn_output = joint[:, :seq_txt, :]
                img_attn_output = joint[:, seq_txt:, :]
                img_attn_output = attn.to_out[0](img_attn_output.contiguous())
                if len(attn.to_out) > 1:
                    img_attn_output = attn.to_out[1](img_attn_output)
                txt_attn_output = attn.to_add_out(txt_attn_output.contiguous())
                return img_attn_output, txt_attn_output

        for blk in pipe.transformer.transformer_blocks:
            blk.attn.set_processor(P())


if __name__ == "__main__":
    run(Model(), argparser().parse_args())
