"""One browsable sample across every family cell, prompt printed under each image.

The per-family sheets in pilot/images/sheets/ are exhaustive and there are 27 of them.
This picks a small representative sample instead — by default one item per family, chosen
so that the S condition is one the judge scored as a failure, because that is the thing
worth looking at. Falls back to the first item if no failure exists in that family.

Usage: python scripts/make_sample_sheet.py [--per-family 1] [--rows-per-page 4]
Writes pilot/images/sheets/SAMPLE_p<n>.png
"""
import argparse
import json
import os
import textwrap

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "pilot", "images")
SHEETS = os.path.join(IMG, "sheets")

CELL = 400
PAD = 16
CAPTION_H = 108
HEADER_H = 34
COND_ORDER = ["S", "S_exp", "S_imp", "P", "A"]
COND_COLOR = {"S": (176, 0, 32), "S_exp": (176, 0, 32), "S_imp": (204, 85, 0),
              "P": (0, 102, 51), "A": (80, 80, 80)}
FONT_DIR = "/usr/share/fonts/truetype/dejavu"

# Order families as the taxonomy does, with the rebuilt versions next to their originals.
FAMILY_ORDER = ["1_existence", "2_attribution", "3_figurative", "4a_occlusion",
                "4a2_occlusion_v2", "4b_legibility", "5_use_mention", "5b_text_bearing",
                "6_relevance", "6b_relevance_v2", "6c_relevance_v3"]

NOTE = {
    "4a_occlusion": "v1 — retired: saturated, and a correct S is indistinguishable from A",
    "4a2_occlusion_v2": "v2 — camera at the observer, S split into explicit / implicit",
    "5_use_mention": "v1 — crate: no occasion to render text at all",
    "5b_text_bearing": "v2 — carriers that must carry text; FLUX writes pseudo-text",
    "6_relevance": "v1 — flagged with 'Unrelated aside:', which marks an unmarked family",
    "6b_relevance_v2": "v2 — transcript form: FLUX draws a comic strip, in A as well",
    "6c_relevance_v3": "v3 — prose form: works",
}


def load_manifest():
    rows = {}
    for fn in sorted(os.listdir(IMG)):
        if fn.startswith("manifest") and fn.endswith(".jsonl"):
            for l in open(os.path.join(IMG, fn)):
                if l.strip():
                    r = json.loads(l)
                    rows.setdefault(r["family"], {}).setdefault(r["id"], {})[r["condition"]] = r
    return rows


def load_verdicts():
    """judge_realized per (id, condition), from the corrected binary protocol."""
    v = {}
    for fam in os.listdir(IMG):
        p = os.path.join(IMG, fam, "results_binary.jsonl")
        if os.path.isfile(p):
            for l in open(p):
                if l.strip():
                    r = json.loads(l)
                    v[(r["id"], r["condition"])] = r["judge_realized"]
    return v


def pick(items, verdicts, n):
    """Prefer items whose S condition the judge scored as realized — the failures."""
    def failed(iid):
        return any(verdicts.get((iid, c)) for c in ("S", "S_imp", "S_exp"))
    ids = sorted(items)
    hits = [i for i in ids if failed(i)]
    return (hits or ids)[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-family", type=int, default=1)
    ap.add_argument("--rows-per-page", type=int, default=4)
    args = ap.parse_args()

    try:
        bold = ImageFont.truetype(os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf"), 15)
        head = ImageFont.truetype(os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf"), 17)
        reg = ImageFont.truetype(os.path.join(FONT_DIR, "DejaVuSans.ttf"), 13)
    except OSError:
        bold = head = reg = ImageFont.load_default()

    manifest, verdicts = load_manifest(), load_verdicts()
    rows = []
    for fam in FAMILY_ORDER:
        if fam not in manifest:
            continue
        for iid in pick(manifest[fam], verdicts, args.per_family):
            rows.append((fam, iid, manifest[fam][iid]))

    os.makedirs(SHEETS, exist_ok=True)
    ncol = max(len([c for c in COND_ORDER if c in conds]) for _, _, conds in rows)
    w = ncol * CELL + PAD * (ncol + 1)
    row_h = HEADER_H + CELL + CAPTION_H + PAD

    out_paths = []
    for page, start in enumerate(range(0, len(rows), args.rows_per_page)):
        chunk = rows[start:start + args.rows_per_page]
        sheet = Image.new("RGB", (w, len(chunk) * row_h + PAD), "white")
        d = ImageDraw.Draw(sheet)
        for r, (fam, iid, conds) in enumerate(chunk):
            y0 = PAD + r * row_h
            title = fam.replace("_", " ")
            if fam in NOTE:
                title += f"   ·   {NOTE[fam]}"
            d.text((PAD, y0), title, fill=(0, 0, 0), font=head)
            present = [c for c in COND_ORDER if c in conds]
            for c, cond in enumerate(present):
                rec = conds[cond]
                x, y = PAD + c * (CELL + PAD), y0 + HEADER_H
                im = Image.open(os.path.join(ROOT, rec["path"])).convert("RGB")
                sheet.paste(im.resize((CELL, CELL), Image.LANCZOS), (x, y))
                d.rectangle([x, y, x + CELL - 1, y + CELL - 1], outline=(205, 205, 205))
                ty = y + CELL + 6
                verdict = verdicts.get((iid, cond))
                tag = "" if verdict is None else ("  → entity present" if verdict
                                                  else "  → entity absent")
                d.text((x, ty), f"[{cond}]{tag}", fill=COND_COLOR[cond], font=bold)
                ty += 19
                for line in textwrap.wrap(rec["prompt"], width=56)[:4]:
                    d.text((x, ty), line, fill=(40, 40, 40), font=reg)
                    ty += 16
        p = os.path.join(SHEETS, f"SAMPLE_p{page+1}.png")
        sheet.save(p)
        out_paths.append(p)
        print(p)
    return out_paths


if __name__ == "__main__":
    main()
