"""Phase 3 — VLM judge over the Phase 2 images.

One structured question per family. Families 1, 4a, 5 and 6 are presence checks;
families 2 and 3 are *marking* checks (is it there as an object, or explicitly
embedded?); family 4b is the geometry check — which way does the inscription face.
Q2 of the pilot is whether the judge can be trusted on 2 and 4b, so those two families
are also inspected by hand and compared in compare_judge.py.

Answers are constrained to a single letter and re-asked once if unparsable.

Usage: CUDA_VISIBLE_DEVICES=5 python scripts/judge_images.py
"""
import json
import os
import re
import time

os.environ.setdefault("HF_HOME", "/data/users/jiahao_huang/hf")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "pilot", "images")

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

PREAMBLE = ("Look at the image and answer with a single capital letter only, "
            "no explanation.\n\n")

YESNO = "Answer with one word, Yes or No, and nothing else.\n\n"

# --- corrected protocol (see scripts/judge_4b_protocols.py, judge_marking_protocols.py) --
# The multiple-choice questions above reach kappa 0.80 / 0.42 / 0.23 against direct
# inspection on families 2 / 3 / 4b. One plain, positive, content-level yes-no question
# reaches 0.94 / 1.00 / 0.94 on the same images. Four wording faults were identified and
# are avoided here:
#   - a third "hedged" option gives an uncertain judge somewhere to park;
#   - listing embedding devices ("a picture, statue, logo…") makes it answer about the
#     medium, since every generated image *is* a picture;
#   - the words "real"/"real live" make it answer about the image's ontological status —
#     asked whether a "real live elephant" is present it answered No on all 72 images;
#   - negatively framed questions are simply agreed with (89% yes-rate against a 42% truth
#     rate when the polarity is flipped).
# These questions were selected after seeing where the originals failed, on the same 108
# inspected images, so they are an in-sample fix and need out-of-sample confirmation.
BINARY_QUESTION = {
    "1_existence": lambda e: f"Is there {article(e)} {e} anywhere in this image?",
    "2_attribution": lambda e: f"Is there {article(e)} {e} anywhere in this image?",
    "3_figurative": lambda e: f"Is there {article(e)} {e} anywhere in this image?",
    "4a_occlusion": lambda e: f"Is any part of {article(e)} {e} visible in this image?",
    "4a2_occlusion_v2": lambda e: f"Is any part of {article(e)} {e} visible in this image?",
    "4b_legibility": lambda e: ("Can you, the viewer of this image, read written words on "
                                "the letter, screen or page that the person is holding or "
                                "looking at?"),
    "5_use_mention": lambda e: f'Is the word "{e.upper()}" written anywhere in this image?',
    "5b_text_bearing": lambda e: f'Is the word "{e.upper()}" written anywhere in this image?',
    "6_relevance": lambda e: f"Is there {article(e)} {e} anywhere in this image?",
    "6b_relevance_v2": lambda e: f"Is there {article(e)} {e} anywhere in this image?",
    "6c_relevance_v3": lambda e: f"Is there {article(e)} {e} anywhere in this image?",
    "4a3_occlusion_v3": lambda e: f"Is any part of {article(e)} {e} visible in this image?",
    "6d_relevance_draw": lambda e: f"Is there {article(e)} {e} anywhere in this image?",
}


def question(row):
    """Return (question_text, {letter: meaning}, realized_letters)."""
    e = row["entity"]
    fam = row["family"]
    if fam in ("1_existence", "6_relevance"):
        return (f"Is there {article(e)} {e} shown in this image?\n"
                f"A. Yes, {article(e)} {e} is present in the scene\n"
                f"B. No {e} is present\n",
                {"A": f"{e} present", "B": "absent"}, {"A"})
    if fam == "2_attribution":
        return (f"How does {article(e)} {e} appear in this image?\n"
                f"A. As an actual {e} present in the scene\n"
                f"B. Only as an explicitly imagined element — inside a thought bubble, "
                f"a dream haze, a framed inset, or a transparent/ghostly overlay\n"
                f"C. No {e} appears anywhere in the image\n",
                {"A": "unmarked realization", "B": "marked", "C": "absent"}, {"A"})
    if fam == "3_figurative":
        return (f"How does {article(e)} {e} appear in this image?\n"
                f"A. As an actual {e} present in the scene as a real animal\n"
                f"B. Only as a visual figure of speech — a picture, statue, logo, pattern, "
                f"costume or shadow shaped like {article(e)} {e}\n"
                f"C. No {e} appears anywhere in the image\n",
                {"A": "unmarked realization", "B": "marked", "C": "absent"}, {"A"})
    if fam == "4a_occlusion":
        return (f"Is any part of {article(e)} {e} visible in this image?\n"
                f"A. Yes, some or all of the {e} is visible\n"
                f"B. No part of any {e} is visible\n",
                {"A": "visible", "B": "hidden/absent"}, {"A"})
    if fam == "4b_legibility":
        return ("The image shows a person with a letter, screen, or notebook page. "
                "Which way does the written text on it face?\n"
                "A. It faces the viewer of the image (the camera), so the viewer can read it\n"
                "B. It faces the person in the scene, away from the viewer, so the viewer "
                "cannot read it\n"
                "C. There is no legible writing on it\n",
                {"A": "faces camera", "B": "faces the reader", "C": "no text"}, {"A"})
    if fam == "5_use_mention":
        return (f"Does this image contain the word \"{e.upper()}\" written out as text "
                f"(printed, stencilled or painted on any surface)?\n"
                f"A. Yes, the word \"{e.upper()}\" is written somewhere in the image\n"
                f"B. No such written word appears\n",
                {"A": "word rendered", "B": "no word"}, {"A"})
    raise ValueError(fam)


def secondary_question(row):
    """Second question for the families where one check is not enough."""
    e = row["entity"]
    if row["family"] == "5_use_mention":
        return (f"Is there {article(e)} actual {e} (the animal) shown in this image?\n"
                f"A. Yes\nB. No\n", "referent_present", {"A"})
    if row["family"] == "4b_legibility":
        return ("Can the person holding or reading it actually see the written side?\n"
                "A. Yes, the written side faces them\n"
                "B. No, the written side faces away from them\n"
                "C. Cannot tell\n", "reader_can_see", {"A"})
    return None


def article(word):
    return "an" if word[0].lower() in "aeiou" else "a"


def main():
    import argparse

    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="smoke-test on the first N images")
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--protocol", choices=["binary", "multichoice"], default="binary",
                    help="binary is the corrected protocol; multichoice reproduces the "
                         "original multiple-choice questions")
    ap.add_argument("--families", default="", help="comma-separated family filter")
    args = ap.parse_args()

    manifest = []
    seen = set()
    for fn in sorted(os.listdir(IMG)):
        if fn.startswith("manifest") and fn.endswith(".jsonl"):
            for l in open(os.path.join(IMG, fn)):
                if l.strip():
                    r = json.loads(l)
                    if (r["id"], r["condition"]) not in seen:
                        seen.add((r["id"], r["condition"]))
                        manifest.append(r)
    manifest.sort(key=lambda r: (r["family"], r["id"], r["condition"]))
    if args.families:
        keep = set(args.families.split(","))
        manifest = [r for r in manifest if r["family"] in keep]
    if args.limit:
        manifest = manifest[:args.limit]
    print(f"{len(manifest)} images to judge", flush=True)

    proc = AutoProcessor.from_pretrained(MODEL)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    def ask(image, text):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PREAMBLE + text}]}]
        chat = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[chat], images=[image], return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        ans = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        m = re.search(r"\b([A-C])\b", ans.strip())
        return (m.group(1) if m else None), ans.strip()

    def ask_binary(image, text):
        msgs = [{"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": YESNO + text}]}]
        chat = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[chat], images=[image], return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=6, do_sample=False)
        raw = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        m = re.search(r"\b(yes|no)\b", raw, flags=re.I)
        return ((m.group(1).lower() == "yes") if m else None), raw.strip()

    out_rows = []
    t0 = time.time()
    for i, row in enumerate(manifest):
        img = Image.open(os.path.join(ROOT, row["path"])).convert("RGB")

        if args.protocol == "binary":
            qtext = BINARY_QUESTION[row["family"]](row["entity"])
            yes, raw = ask_binary(img, qtext)
            rec = dict(row)
            rec.update({"judge_protocol": "binary", "judge_question": qtext,
                        "judge_raw": raw, "judge_realized": yes,
                        "judge_letter": None if yes is None else ("A" if yes else "B"),
                        "judge_option_map": {"A": "realized", "B": "not realized"},
                        "judge_meaning": None if yes is None else ("realized" if yes else "not realized")})
            out_rows.append(rec)
            if i % 25 == 0:
                print(f"[{i+1}/{len(manifest)}] {row['id']}_{row['condition']} -> {raw!r}", flush=True)
            continue

        qtext, meanings, realized_letters = question(row)
        letter, raw = ask(img, qtext)
        rec = dict(row)
        rec.update({"judge_protocol": "multichoice", "judge_question": qtext,
                    "judge_letter": letter, "judge_raw": raw,
                    "judge_meaning": meanings.get(letter), "judge_option_map": meanings,
                    "judge_realized": (letter in realized_letters) if letter else None})
        sec = secondary_question(row)
        if sec:
            stext, skey, sletters = sec
            sletter, sraw = ask(img, stext)
            rec.update({f"judge2_{skey}": (sletter in sletters) if sletter else None,
                        "judge2_letter": sletter, "judge2_raw": sraw, "judge2_question": stext})
        out_rows.append(rec)
        if i % 25 == 0:
            print(f"[{i+1}/{len(manifest)}] {row['id']}_{row['condition']} -> {letter}", flush=True)

    by_family = {}
    for r in out_rows:
        by_family.setdefault(r["family"], []).append(r)
    for fam, rows in by_family.items():
        d = os.path.join(IMG, fam)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"results{args.out_suffix}.jsonl"), "w") as f:
            for r in sorted(rows, key=lambda x: (x["id"], x["condition"])):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"judged {len(out_rows)} images in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
