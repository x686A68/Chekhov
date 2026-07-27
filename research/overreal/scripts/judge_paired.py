"""Phase 3b — the paired forced-choice judge protocol.

`overrealization.tex` P2 designates paired forced choice as the *primary* evaluation
protocol: show the judge the S and P outputs together and ask which came from which
prompt, because relative judgements beat absolute ones and pairing neutralises a judge
biased toward one answer. The absolute per-image protocol in judge_images.py is what
Q2 was answered with, so this script re-runs families 2, 3 and 4b under the protocol the
paper actually plans to use.

The S and P images of an item are tiled side by side (left/right order decided by a
stable hash of the item id, so the judge cannot exploit position) and the judge picks the
side that came from the licensing prompt. Ground truth is known by construction.

Writes pilot/images/paired_forced_choice.json.
"""
import json
import os
import re
import zlib

os.environ.setdefault("HF_HOME", "/data/users/jiahao_huang/hf")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "pilot", "images")
MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

FAMILIES = ["2_attribution", "3_figurative", "4b_legibility"]

PREAMBLE = ("The two images below are shown side by side: image 1 on the left, image 2 on "
            "the right. Answer with a single digit, 1 or 2, and nothing else.\n\n")


def question(row):
    e = row["entity"]
    art = "an" if e[0].lower() in "aeiou" else "a"
    if row["family"] == "2_attribution":
        return (f"One of these two images was generated from a prompt stating that {art} {e} "
                f"was really there. The other came from a prompt saying only that someone "
                f"*believed* {art} {e} was there — in that one, no {e} should actually be in "
                f"the scene.\n\nIn which image is {art} {e} actually present in the scene?")
    if row["family"] == "3_figurative":
        return (f"One of these two images was generated from a prompt placing {art} real {e} "
                f"beside the person. The other came from a prompt that only compared the "
                f"person to {art} {e} as a figure of speech — in that one, no {e} should "
                f"actually appear.\n\nIn which image is {art} real {e} present in the scene?")
    if row["family"] == "4b_legibility":
        return ("One of these two images was generated from a prompt in which the writing is "
                "held up toward the camera. The other came from a prompt in which the person "
                "is reading it to herself, so the writing should face her and away from the "
                "camera.\n\nIn which image can the viewer read the writing?")
    raise ValueError(row["family"])


def main():
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    manifest = {}
    for fn in sorted(os.listdir(IMG)):
        if fn.startswith("manifest") and fn.endswith(".jsonl"):
            for l in open(os.path.join(IMG, fn)):
                if l.strip():
                    r = json.loads(l)
                    manifest[(r["id"], r["condition"])] = r

    human = {}
    hp = os.path.join(IMG, "inspection.jsonl")
    if os.path.exists(hp):
        for l in open(hp):
            if l.strip():
                r = json.loads(l)
                human[(r["id"], r["condition"])] = r["human_letter"]

    proc = AutoProcessor.from_pretrained(MODEL)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    results = []
    for fam in FAMILIES:
        ids = sorted({k[0] for k in manifest if k[0].startswith(fam)})
        for iid in ids:
            s, p = manifest.get((iid, "S")), manifest.get((iid, "P"))
            if not s or not p:
                continue
            p_on_left = zlib.crc32(iid.encode()) % 2 == 0
            left, right = (p, s) if p_on_left else (s, p)
            li = Image.open(os.path.join(ROOT, left["path"])).convert("RGB").resize((512, 512))
            ri = Image.open(os.path.join(ROOT, right["path"])).convert("RGB").resize((512, 512))
            pair = Image.new("RGB", (1024, 512), "white")
            pair.paste(li, (0, 0))
            pair.paste(ri, (512, 0))

            q = PREAMBLE + question(s)
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
            chat = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = proc(text=[chat], images=[pair], return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=6, do_sample=False)
            raw = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                    skip_special_tokens=True)[0].strip()
            m = re.search(r"[12]", raw)
            picked = int(m.group()) if m else None
            picked_cond = None
            if picked:
                picked_cond = ("P" if p_on_left else "S") if picked == 1 else ("S" if p_on_left else "P")

            # A pair is only informative if the two images actually differ on the scored
            # dimension; when the model over-realizes under S they are indistinguishable
            # and no answer can be right.
            hs, hp_ = human.get((iid, "S")), human.get((iid, "P"))
            discriminable = (hs is not None and hp_ is not None and hs != hp_)
            results.append({
                "id": iid, "family": fam, "entity": s["entity"], "p_on_left": p_on_left,
                "judge_raw": raw, "picked_side": picked, "picked_condition": picked_cond,
                "correct": (picked_cond == "P") if picked_cond else None,
                "human_S": hs, "human_P": hp_, "discriminable_by_inspection": discriminable,
            })
            print(f"{iid} p_left={p_on_left} -> {raw!r} = {picked_cond}", flush=True)

    summary = {}
    for fam in FAMILIES:
        rows = [r for r in results if r["family"] == fam]
        disc = [r for r in rows if r["discriminable_by_inspection"]]
        def acc(rs):
            got = [r for r in rs if r["correct"] is not None]
            return round(sum(r["correct"] for r in got) / len(got), 4) if got else None
        summary[fam] = {
            "n_pairs": len(rows), "accuracy_all_pairs": acc(rows),
            "n_discriminable": len(disc), "accuracy_discriminable": acc(disc),
            "n_indistinguishable": len(rows) - len(disc),
            "side_bias_left": round(sum(1 for r in rows if r["picked_side"] == 1) / len(rows), 4),
        }
    with open(os.path.join(IMG, "paired_forced_choice.json"), "w") as f:
        json.dump({"summary": summary, "pairs": results}, f, indent=2)

    print()
    for fam, s in summary.items():
        print(f"{fam:<16} all={s['accuracy_all_pairs']} (n={s['n_pairs']})  "
              f"discriminable={s['accuracy_discriminable']} (n={s['n_discriminable']})  "
              f"left-bias={s['side_bias_left']}")


if __name__ == "__main__":
    main()
