"""Phase 3d — the same protocol fix applied to the marking families (2 and 3).

judge_4b_protocols.py found that the three-way question was the problem, not the model:
a single positively framed binary reached kappa 0.94 where the three-way form reached
0.63, and a negatively framed binary collapsed to 0.30 with the judge agreeing to
whatever the question presupposed 89% of the time.

Families 2 and 3 failed the same way — the judge over-used option B, the hedged
"it is only imagined / only a picture" option. This tests whether decomposing into two
positive binaries fixes them too:

  Q1  is there a real E in the scene at all?          (presence)
  Q2  is it shown inside a thought bubble / picture?   (embedding)

  unmarked realization = Q1 yes AND Q2 no

Writes pilot/images/marking_protocols.json.
"""
import json
import os
import re

os.environ.setdefault("HF_HOME", "/data/users/jiahao_huang/hf")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "pilot", "images")
MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
FAMILIES = ["2_attribution", "3_figurative"]

YESNO = "Answer with one word, Yes or No, and nothing else.\n\n"


def article(w):
    return "an" if w[0].lower() in "aeiou" else "a"


def q1(e):
    return YESNO + f"Is there {article(e)} {e} anywhere in this image?"


def q2(e, fam):
    """Second step of the chain. Note the failure this wording caused: listing the
    embedding devices invites the model to answer with one of the listed words
    ("Picture", "Statue") instead of yes/no, and — worse — to pick one for images that
    plainly contain a real animal. Kept as-is because that behaviour is the finding;
    q_single below is the corrected form."""
    if fam == "2_attribution":
        return (YESNO + f"The {e} in this image: is it drawn inside a thought bubble, a dream "
                f"cloud, a framed inset, or as a transparent ghostly overlay?")
    return (YESNO + f"The {e} in this image: is it a picture, statue, logo, pattern, costume "
            f"or shadow rather than a real live {e} standing in the scene?")


def q_single(e):
    """One positive, unlisted, directly perceptual question — the form that reached
    kappa 0.94 on family 4b."""
    return YESNO + f"Is there a real live {e} standing in the scene in this image?"


def main():
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    rows = []
    for fam in FAMILIES:
        p = os.path.join(IMG, fam, "results.jsonl")
        rows += [r for r in (json.loads(l) for l in open(p) if l.strip()) if r.get("human_letter")]
    print(f"{len(rows)} inspected images across {FAMILIES}", flush=True)

    proc = AutoProcessor.from_pretrained(MODEL)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    def ask(image, text):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        chat = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[chat], images=[image], return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=6, do_sample=False)
        raw = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        m = re.search(r"\b(yes|no)\b", raw, flags=re.I)
        return (m.group(1).lower() == "yes") if m else None, raw.strip()

    out_rows = []
    for i, r in enumerate(rows):
        img = Image.open(os.path.join(ROOT, r["path"])).convert("RGB")
        e = r["entity"]
        present, raw1 = ask(img, q1(e))
        embedded, raw2 = (None, "") if not present else ask(img, q2(e, r["family"]))
        # the listed-options wording gets answered with an option word rather than
        # yes/no; any such answer means "yes, it is embedded"
        if present and embedded is None and raw2:
            embedded = True
        chain = bool(present and embedded is False)
        single, raw3 = ask(img, q_single(e))
        out_rows.append({
            "single_unmarked": single, "single_raw": raw3,
            "id": r["id"], "family": r["family"], "condition": r["condition"], "entity": e,
            "human_letter": r["human_letter"],
            "truth_unmarked": r["human_letter"] == "A",
            "chain_unmarked": chain, "q1_present": present, "q2_embedded": embedded,
            "q1_raw": raw1, "q2_raw": raw2,
            "v0_threeway": r["judge_letter"] == "A", "v0_letter": r["judge_letter"],
        })
        if i % 20 == 0:
            print(f"[{i+1}/{len(rows)}] {r['id']}_{r['condition']}", flush=True)

    def kappa(a, b):
        n = len(a)
        po = sum(x == y for x, y in zip(a, b)) / n
        pe = sum((sum(1 for x in a if x == v) / n) * (sum(1 for y in b if y == v) / n)
                 for v in (True, False))
        return None if pe == 1.0 else round((po - pe) / (1 - pe), 4)

    summary = {}
    for fam in FAMILIES:
        sel = [r for r in out_rows if r["family"] == fam]
        truth = [r["truth_unmarked"] for r in sel]
        entry = {"n": len(sel), "truth_unmarked_rate": round(sum(truth) / len(sel), 4)}
        for v in ("v0_threeway", "chain_unmarked", "single_unmarked"):
            pred = [r[v] for r in sel]
            entry[v] = {"accuracy": round(sum(p == t for p, t in zip(pred, truth)) / len(sel), 4),
                        "cohens_kappa": kappa(pred, truth),
                        "says_unmarked_rate": round(sum(pred) / len(sel), 4)}
        # how often the two-step chain stops at Q1 vs is overruled by Q2
        entry["q2_overruled_q1"] = sum(1 for r in sel if r["q1_present"] and r["q2_embedded"])
        summary[fam] = entry

    with open(os.path.join(IMG, "marking_protocols.json"), "w") as f:
        json.dump({"summary": summary, "rows": out_rows}, f, indent=2)

    print()
    for fam, s in summary.items():
        print(f"--- {fam}  (truth unmarked rate {s['truth_unmarked_rate']}) ---")
        for v in ("v0_threeway", "chain_unmarked", "single_unmarked"):
            d = s[v]
            print(f"  {v:<16} acc={d['accuracy']:<8} kappa={str(d['cohens_kappa']):<8} "
                  f"says-unmarked={d['says_unmarked_rate']}")
        print(f"  Q2 overruled Q1 on {s['q2_overruled_q1']} images")


if __name__ == "__main__":
    main()
