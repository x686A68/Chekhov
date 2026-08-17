"""Presence judge over a Family 1 image directory (nb2_pilot / nb2_full manifests).

Uses the corrected binary protocol from judge_images.py: one plain, positively framed
content question, Yes/No only. On the pilot's hand-inspected images that protocol
reached kappa 0.94-1.00 where the original multiple-choice questions reached 0.23-0.80
(pilot/REPORT.md 4.2-4.4). The question here is verbatim the family-1 entry:

    "Is there a/an {target} anywhere in this image?"

Yes => the target was realized => over-realization (the prompt asked for its absence).

Usage:
    CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/judge_family1_batch.py \
        --dir dataset/family1/images/nb2_full
"""
import argparse
import json
import os
import re
import time

os.environ.setdefault("HF_HOME", "/data/users/jiahao_huang/hf")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
YESNO = "Answer with one word, Yes or No, and nothing else.\n\n"


def article(word):
    return "an" if word[0].lower() in "aeiou" else "a"


def question(target):
    return f"Is there {article(target)} {target} anywhere in this image?"


def main():
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="image dir containing manifest.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="judgements.jsonl")
    args = ap.parse_args()

    d = args.dir if os.path.isabs(args.dir) else os.path.join(ROOT, args.dir)
    rows = [json.loads(l) for l in open(os.path.join(d, "manifest.jsonl"))]
    rows = [r for r in rows if r["status"] == "ok" and r.get("image_path")]
    rows.sort(key=lambda r: (r["item_id"], r["sample"]))
    if args.limit:
        rows = rows[:args.limit]
    print(f"{len(rows)} images to judge", flush=True)

    proc = AutoProcessor.from_pretrained(MODEL)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    def ask(image, text):
        msgs = [{"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": YESNO + text}]}]
        chat = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[chat], images=[image], return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=6, do_sample=False)
        raw = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True)[0]
        m = re.search(r"\b(yes|no)\b", raw, flags=re.I)
        return ((m.group(1).lower() == "yes") if m else None), raw.strip()

    out_path = os.path.join(d, args.out)
    t0 = time.time()
    n_real = n_none = 0
    with open(out_path, "w") as f:
        for i, r in enumerate(rows):
            p = r["image_path"]
            img = Image.open(p if os.path.isabs(p) else os.path.join(ROOT, p)).convert("RGB")
            q = question(r["target"])
            yes, raw = ask(img, q)
            n_real += yes is True
            n_none += yes is None
            rec = dict(r)
            rec.update(judge_model=MODEL, judge_protocol="binary", judge_question=q,
                       judge_raw=raw, judge_realized=yes)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if i % 50 == 0:
                el = time.time() - t0
                print(f"[{i+1}/{len(rows)}] {r['item_id']}_s{r['sample']} "
                      f"{r['target']} -> {raw!r} | realized so far {n_real} "
                      f"| {el:.0f}s", flush=True)
    print(f"done: {len(rows)} judged, {n_real} realized, {n_none} unparsable "
          f"-> {out_path}", flush=True)


if __name__ == "__main__":
    main()
