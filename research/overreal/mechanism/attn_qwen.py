"""Attention recorder for Qwen-Image (see attn_common.py).

Joint sequence: [text, image (4096)]. True CFG runs the conditional and the
negative-prompt pass separately each step; only calls whose text length
equals the conditional prompt's are recorded. Text tokens are the Qwen2.5-VL
tokens of the templated prompt after the 34 dropped template tokens (the
trailing template tokens stay in the sequence, as in the pipeline).

Usage: CUDA_VISIBLE_DEVICES=0 python attn_qwen.py --shard 0/6
"""
import os

os.environ.setdefault("HF_HOME", "/data/users/jiahao_huang/hf")

import torch

from attn_common import argparser, run, span_token_indices

MODEL = "Qwen/Qwen-Image"
SIZE = 1024
TEMPLATE = ("<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
            "texture, quantity, text, spatial relationships of the objects and background:"
            "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n")
DROP = 34


class Model:
    key = "qwen-image"
    steps_default, cfg_default = 50, 4.0
    n_blocks = 60
    n_img = (SIZE // 16) ** 2

    def load(self):
        from diffusers import DiffusionPipeline
        return DiffusionPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda")

    def _enc(self, pipe, text):
        return pipe.tokenizer(TEMPLATE.format(text), return_offsets_mapping=True)

    def n_txt(self, pipe, text):
        return len(self._enc(pipe, text)["input_ids"]) - DROP

    def tokens(self, pipe, text, spans):
        enc = self._enc(pipe, text)
        prefix = TEMPLATE.index("{}")
        offs = enc["offset_mapping"][DROP:]
        # only tokens inside the user prompt count as real
        real = [(a, b) if (b > a and a >= prefix and b <= prefix + len(text)) else (0, 0) for a, b in offs]
        return span_token_indices(real, spans, prefix)

    def install(self, pipe, rec):
        from diffusers.models.transformers.transformer_qwenimage import QwenDoubleStreamAttnProcessor2_0
        n_txt_cond = rec.n_txt

        class P(QwenDoubleStreamAttnProcessor2_0):
            def __init__(self, block):
                super().__init__()
                self.block = block

            def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                         encoder_hidden_states_mask=None, attention_mask=None, image_rotary_emb=None):
                from diffusers.models.transformers.transformer_qwenimage import ROPE_PER_DEVICE
                from diffusers.models.attention_dispatch import dispatch_attention_fn
                if encoder_hidden_states_mask is not None:
                    seq_img = hidden_states.shape[1]
                    image_mask = torch.ones((hidden_states.shape[0], seq_img), dtype=torch.bool,
                                            device=hidden_states.device)
                    attention_mask = torch.cat([encoder_hidden_states_mask, image_mask], dim=1)
                    attention_mask = attention_mask[:, None, None, :]
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
                joint_query = torch.cat([txt_query, img_query], dim=1)
                joint_key = torch.cat([txt_key, img_key], dim=1)
                joint_value = torch.cat([txt_value, img_value], dim=1)
                if seq_txt == n_txt_cond:
                    n_seq = joint_query.shape[1]
                    rec.record(self.block, joint_query[0].transpose(0, 1), joint_key[0].transpose(0, 1),
                               txt=slice(0, seq_txt), img=slice(seq_txt, n_seq))
                joint_hidden_states = dispatch_attention_fn(
                    joint_query, joint_key, joint_value, attn_mask=attention_mask, dropout_p=0.0,
                    is_causal=False, backend=self._attention_backend, parallel_config=self._parallel_config)
                joint_hidden_states = joint_hidden_states.flatten(2, 3).to(joint_query.dtype)
                txt_attn_output = joint_hidden_states[:, :seq_txt, :]
                img_attn_output = joint_hidden_states[:, seq_txt:, :]
                img_attn_output = attn.to_out[0](img_attn_output.contiguous())
                if len(attn.to_out) > 1:
                    img_attn_output = attn.to_out[1](img_attn_output)
                txt_attn_output = attn.to_add_out(txt_attn_output.contiguous())
                return img_attn_output, txt_attn_output

        for i, blk in enumerate(pipe.transformer.transformer_blocks):
            blk.attn.set_processor(P(i))

    def generate(self, pipe, text, steps, cfg, seed):
        gen = torch.Generator("cuda").manual_seed(seed)
        return pipe(prompt=text, negative_prompt=" ", true_cfg_scale=cfg, num_inference_steps=steps,
                    width=SIZE, height=SIZE, generator=gen).images[0]


if __name__ == "__main__":
    run(Model(), argparser().parse_args())
