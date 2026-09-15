"""Ask Qwen-Image's text encoder directly. The encoder is Qwen2.5-VL-7B, an
instruction model, so the same weights that condition Qwen-Image can be asked
whether the target would be visible. Expected answers: S -> no, P -> yes.

One uniform question for all four families, matching the paper's definition
(the target should not appear in the image):
  "If this prompt were rendered as an image exactly as described, would a
   <target> be visible in the image? Answer yes or no."

Output: results/ask_encoder.jsonl (one line per item and side, with the raw
answer) and a per-family accuracy table on stdout.

Usage: CUDA_VISIBLE_DEVICES=4 python ask_encoder.py
"""
import json
import os

os.environ.setdefault("HF_HOME", "/data/users/jiahao_huang/hf")

import torch
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
PAIRS = os.path.join(ROOT, "pairs.jsonl")
OUT = os.path.join(ROOT, "results", "ask_encoder.jsonl")
QWEN_IMAGE = "Qwen/Qwen-Image"

QUESTION = ("Here is a text-to-image prompt:\n\"{prompt}\"\n\n"
            "If this prompt were rendered as an image exactly as described, "
            "would {target} be visible in the image? Answer yes or no.")


def article(t):
    t = t.strip().strip('"').strip("'").rstrip(".").strip()
    if t.lower().startswith(("a ", "an ", "the ")):
        return t
    return ("an " if t[0].lower() in "aeiou" else "a ") + t


def main():
    from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration
    tok = AutoTokenizer.from_pretrained(QWEN_IMAGE, subfolder="tokenizer")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_IMAGE, subfolder="text_encoder", torch_dtype=torch.bfloat16).to("cuda").eval()
    tok.padding_side = "left"

    rows = [r for r in (json.loads(l) for l in open(PAIRS)) if not r["exclude"]]
    jobs = []
    for r in rows:
        for side in ("S", "P"):
            text = r["s_text"] if side == "S" else r["p_text"]
            msgs = [{"role": "user", "content": QUESTION.format(prompt=text, target=article(r["target"]))}]
            jobs.append((r, side, tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out = open(OUT, "w")
    acc = defaultdict(lambda: [0, 0])
    bs = 16
    for i in range(0, len(jobs), bs):
        chunk = jobs[i:i + bs]
        enc = tok([j[2] for j in chunk], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=4, do_sample=False)
        answers = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for (r, side, _), ans in zip(chunk, answers):
            a = ans.strip().lower()
            yes = a.startswith("yes")
            no = a.startswith("no")
            expected = "no" if side == "S" else "yes"
            correct = (yes and expected == "yes") or (no and expected == "no")
            acc[(r["family"], side)][0] += int(correct)
            acc[(r["family"], side)][1] += 1
            out.write(json.dumps({"item_id": r["item_id"], "family": r["family"], "side": side,
                                  "target": r["target"], "answer": ans.strip(),
                                  "expected": expected, "correct": bool(correct)}) + "\n")
        print(f"{min(i + bs, len(jobs))}/{len(jobs)}", flush=True)
    out.close()

    print(f"{'family':<14}{'S: no':>10}{'P: yes':>10}{'both':>10}")
    for fam in ["cancellation", "attribution", "figurative", "perspectival"]:
        s = acc[(fam, "S")]
        p = acc[(fam, "P")]
        print(f"{fam:<14}{s[0]/max(s[1],1):>10.2f}{p[0]/max(p[1],1):>10.2f}"
              f"{(s[0]+p[0])/max(s[1]+p[1],1):>10.2f}")


if __name__ == "__main__":
    main()
