"""Qwen official prompt expansion, run locally.

Uses the verbatim SYSTEM_PROMPT and magic-suffix concatenation from the
Qwen-Image repo's tools/prompt_utils.py (vendored next to this script as
qwen_official_prompt_utils.py), but substitutes local Qwen3-32B for the
DashScope qwen-plus call, so the expansion is reproducible and free.
The substitution is the only deviation from the official pipeline and is
recorded in each output row.

Usage: CUDA_VISIBLE_DEVICES=0 python expand_qwen.py [--model Qwen/Qwen3-32B]
"""
import argparse
import datetime
import os
import re

os.environ.setdefault("HF_HOME", "/data/users/jiahao_huang/hf")

from common import EXPANDED, append_jsonl, read_jsonl, load_prompts

MAGIC = "Ultra HD, 4K, cinematic composition"   # appended verbatim upstream


def official_system_prompt():
    src = open(os.path.join(os.path.dirname(__file__),
                            "qwen_official_prompt_utils.py"), encoding="utf-8").read()
    fn = src.split("def polish_prompt_en", 1)[1]
    return re.search(r"SYSTEM_PROMPT = '''(.*?)'''", fn, re.S).group(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto")

    system = official_system_prompt()
    done = {r["item_id"] for r in read_jsonl(EXPANDED)} if EXPANDED.exists() else set()
    todo = [(i, f, p) for i, f, p in load_prompts("raw") if i not in done]
    print(f"{len(todo)} prompts to expand ({len(done)} already done)")

    for n, (item_id, family, prompt) in enumerate(todo, 1):
        # upstream sends the whole template as a single user turn
        text = tok.apply_chat_template(
            [{"role": "system", "content": "You are a helpful assistant."},
             {"role": "user",
              "content": f"{system}\n\nUser Input: {prompt.strip()}\n\n Rewritten Prompt:"}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tok(text, return_tensors="pt").to(model.device)
        out = model.generate(**ids, max_new_tokens=512, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        polished = tok.decode(out[0][ids["input_ids"].shape[1]:],
                              skip_special_tokens=True).strip().replace("\n", " ")
        append_jsonl(EXPANDED, {
            "item_id": item_id,
            "family": family,
            "expander": "qwen",
            "expander_model": f"{args.model} (local; substitutes qwen-plus of the official tool)",
            "text": polished + MAGIC,   # upstream concatenates with no separator
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        })
        if n % 25 == 0 or n == len(todo):
            print(f"[{n}/{len(todo)}] {item_id}")


if __name__ == "__main__":
    main()
