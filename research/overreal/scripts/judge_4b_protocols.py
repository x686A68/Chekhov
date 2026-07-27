"""Phase 3c — protocol variants for family 4b, the judgement the pilot could not score.

The absolute three-way question reached kappa 0.23 against direct inspection, and the
failure was systematic rather than noisy: the judge never once used option C ("no legible
writing"), so it always committed to a facing direction even for a dark laptop lid. The
hypothesis is that the three-way form lets an uncertain judge park its uncertainty in an
option, and that binary decomposition removes the parking space.

Variants, all scored against the same 36 hand-inspected images:

  v0_threeway   the original question (baseline)
  v1_chain      two binaries: is any writing legible at all, then does it face the camera
  v2_single     one binary: can the viewer read the writing
  v3_reversed   the same binary with the polarity flipped, to expose yes-bias
  v4_reasoned   the original question but allowed one sentence of reasoning first

Everything is reduced to the same decision — "the viewer can read the writing", which is
the over-realization — so the variants are directly comparable, and the baseline is
re-scored as a binary too rather than compared across different label spaces.

Writes pilot/images/4b_protocols.json.
"""
import json
import os
import re

os.environ.setdefault("HF_HOME", "/data/users/jiahao_huang/hf")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "pilot", "images")
MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
FAMILY = "4b_legibility"

LETTER_ONLY = "Answer with a single capital letter only, no explanation.\n\n"
YESNO_ONLY = "Answer with one word, Yes or No, and nothing else.\n\n"

V0 = (LETTER_ONLY +
      "The image shows a person with a letter, screen, or notebook page. Which way does "
      "the written text on it face?\n"
      "A. It faces the viewer of the image (the camera), so the viewer can read it\n"
      "B. It faces the person in the scene, away from the viewer, so the viewer cannot read it\n"
      "C. There is no legible writing on it\n")

V1_Q1 = (YESNO_ONLY +
         "Look at the letter, screen or page in this image. Is there any writing on it that "
         "is legible to you — actual readable words, not blurred or blank?")
V1_Q2 = (YESNO_ONLY +
         "The writing on the letter, screen or page: is it turned toward the camera, so that "
         "you the viewer can read it?")

V2 = (YESNO_ONLY +
      "Can you, the viewer of this image, read written words on the letter, screen or page "
      "that the person is holding or looking at?")

V3 = (YESNO_ONLY +
      "Is the written side of the letter, screen or page turned away from the camera, so "
      "that you the viewer cannot read what it says?")

V4 = ("First give one short sentence describing what you see on the letter, screen or page "
      "and which way it faces. Then on a new line answer with a single capital letter.\n\n"
      "Which way does the written text face?\n"
      "A. It faces the viewer of the image (the camera), so the viewer can read it\n"
      "B. It faces the person in the scene, away from the viewer, so the viewer cannot read it\n"
      "C. There is no legible writing on it\n")


def main():
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    rows = [json.loads(l) for l in open(os.path.join(IMG, FAMILY, "results.jsonl")) if l.strip()]
    rows = [r for r in rows if r.get("human_letter")]
    print(f"{len(rows)} inspected {FAMILY} images", flush=True)

    proc = AutoProcessor.from_pretrained(MODEL)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    def ask(image, text, max_new_tokens=8):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        chat = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[chat], images=[image], return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return proc.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                 skip_special_tokens=True)[0].strip()

    def yes(raw):
        m = re.search(r"\b(yes|no)\b", raw, flags=re.I)
        return m.group(1).lower() == "yes" if m else None

    def letter(raw):
        m = re.findall(r"\b([A-C])\b", raw)
        return m[-1] if m else None  # v4 reasons first, so take the last letter

    out_rows = []
    for i, r in enumerate(rows):
        img = Image.open(os.path.join(ROOT, r["path"])).convert("RGB")
        truth = r["human_letter"] == "A"  # the viewer can read it = over-realization

        l0 = letter(ask(img, V0))
        q1, q2 = yes(ask(img, V1_Q1)), None
        if q1:
            q2 = yes(ask(img, V1_Q2))
        v2 = yes(ask(img, V2))
        v3 = yes(ask(img, V3))
        raw4 = ask(img, V4, max_new_tokens=90)
        l4 = letter(raw4)

        out_rows.append({
            "id": r["id"], "condition": r["condition"], "human_letter": r["human_letter"],
            "truth_viewer_can_read": truth,
            "v0_threeway": (l0 == "A") if l0 else None, "v0_letter": l0,
            "v1_chain": bool(q1 and q2), "v1_q1_legible": q1, "v1_q2_faces_camera": q2,
            "v2_single": v2,
            "v3_reversed": (not v3) if v3 is not None else None, "v3_raw_yes": v3,
            "v4_reasoned": (l4 == "A") if l4 else None, "v4_letter": l4, "v4_raw": raw4,
        })
        if i % 10 == 0:
            print(f"[{i+1}/{len(rows)}] {r['id']}_{r['condition']}", flush=True)

    def kappa(a, b):
        n = len(a)
        po = sum(x == y for x, y in zip(a, b)) / n
        pe = sum((sum(1 for x in a if x == v) / n) * (sum(1 for y in b if y == v) / n)
                 for v in (True, False))
        return None if pe == 1.0 else round((po - pe) / (1 - pe), 4)

    truth = [r["truth_viewer_can_read"] for r in out_rows]
    summary = {}
    for v in ("v0_threeway", "v1_chain", "v2_single", "v3_reversed", "v4_reasoned"):
        pred = [r[v] for r in out_rows]
        ok = [(p, t) for p, t in zip(pred, truth) if p is not None]
        acc = round(sum(p == t for p, t in ok) / len(ok), 4) if ok else None
        summary[v] = {
            "n_answered": len(ok), "accuracy": acc,
            "cohens_kappa": kappa([p for p, _ in ok], [t for _, t in ok]),
            "says_readable_rate": round(sum(1 for p, _ in ok if p) / len(ok), 4) if ok else None,
        }
    summary["_truth_readable_rate"] = round(sum(truth) / len(truth), 4)

    with open(os.path.join(IMG, "4b_protocols.json"), "w") as f:
        json.dump({"summary": summary, "rows": out_rows}, f, indent=2)

    print(f"\ntruth: viewer can read in {summary['_truth_readable_rate']:.2f} of images\n")
    print(f"{'variant':<14}{'acc':>8}{'kappa':>9}{'says-readable':>15}")
    for v, s in summary.items():
        if v.startswith("_"):
            continue
        print(f"{v:<14}{s['accuracy']:>8}{str(s['cohens_kappa']):>9}{s['says_readable_rate']:>15}")


if __name__ == "__main__":
    main()
