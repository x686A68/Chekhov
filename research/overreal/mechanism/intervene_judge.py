"""Presence judge for the intervention images, local Qwen2.5-VL-7B (no API).

For every image under intervene/<model>/<cond>/ and the matching baseline
image of data/generation/images/<model>/raw/, ask whether the target is
visible and record yes/no. Text targets (perspectival writing) get a
readability question. Output: intervene/<model>/judge.jsonl (resumable).
Usage: CUDA_VISIBLE_DEVICES=0 python intervene_judge.py --model sd35l [--shard 0/2]
"""
import argparse
import glob
import json
import os
import re

os.environ.setdefault("HF_HOME", "/data/users/jiahao_huang/hf")
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(ROOT, "..", "..", ".."))
VLM = "Qwen/Qwen2.5-VL-7B-Instruct"
TEXTY = re.compile(r"\b(says?|reads?|written|writing|text|words?|letters?|sign|label|printed|note)\b", re.I)


def question(prompt, target):
    if TEXTY.search(prompt) and (target[:1].isupper() or " " in target) and target.lower() not in prompt.lower().split():
        return f'Is the text "{target}" readable anywhere in this image? Answer yes or no.'
    art = "an" if target[:1].lower() in "aeiou" else "a"
    return f"Is there {art} {target} visible anywhere in this image? Answer yes or no."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args()
    pairs = {json.loads(l)["item_id"]: json.loads(l) for l in open(os.path.join(ROOT, "pairs.jsonl"))}
    out_dir = os.path.join(ROOT, "intervene", args.model)
    jobs = []
    for fn in sorted(glob.glob(os.path.join(out_dir, "*", "*.jpg"))):
        cond = os.path.basename(os.path.dirname(fn)); stem = os.path.basename(fn)[:-4]
        item, seed = stem.rsplit("__s", 1); item = item.replace("_", "/", 1)
        jobs.append((cond, item, int(seed), fn))
    # baselines for the same (item, seed)
    seen = set()
    for cond, item, seed, fn in list(jobs):
        if (item, seed) in seen: continue
        seen.add((item, seed))
        fam, pid = item.split("/"); num = pid.split("_")[1]
        base = os.path.join(REPO, "data", "generation", "images", args.model, "raw", f"{fam}_{num}__s{seed}.png")
        if os.path.exists(base): jobs.append(("baseline", item, seed, base))
        pbase = os.path.join(ROOT, "attn", args.model, f"{item.replace('/', '_')}__P_s{seed}.png")
        if os.path.exists(pbase): jobs.append(("P_baseline", item, seed, pbase))
    i, n = map(int, args.shard.split("/")); jobs = jobs[i::n]
    out_path = os.path.join(out_dir, f"judge_{i}of{n}.jsonl")
    done = {(json.loads(l)["cond"], json.loads(l)["item_id"], json.loads(l)["seed"]) for l in open(out_path)} if os.path.exists(out_path) else set()
    jobs = [j for j in jobs if (j[0], j[1], j[2]) not in done]
    print(f"{len(jobs)} images to judge", flush=True)
    if not jobs: return
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    proc = AutoProcessor.from_pretrained(VLM)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(VLM, torch_dtype=torch.bfloat16, device_map="cuda")
    for k, (cond, item, seed, fn) in enumerate(jobs, 1):
        r = pairs[item]; side = "P" if cond.startswith("P_") else "S"
        prompt = r["p_text"] if side == "P" else r["s_text"]
        q = question(prompt, r["target"])
        img = Image.open(fn).convert("RGB"); img.thumbnail((768, 768))
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
        chat = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[chat], images=[img], return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=4, do_sample=False)
        ans = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip().lower()
        present = ans.startswith("yes")
        with open(out_path, "a") as f:
            f.write(json.dumps(dict(cond=cond, item_id=item, family=r["family"], seed=seed, side=side,
                                    target=r["target"], q=q, answer=ans, present=present, file=fn)) + "\n")
        if k % 100 == 0: print(f"[{k}/{len(jobs)}]", flush=True)


if __name__ == "__main__":
    main()
