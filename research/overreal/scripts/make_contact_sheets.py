"""Build contact sheets for direct inspection.

Reading 250 full-resolution PNGs one at a time is not practical, so images are tiled
into labelled sheets: one row per item, one column per condition, each cell captioned
with the item id, the condition and the prompt that produced it, so a sheet can be read
without cross-referencing the manifest.

Usage: python scripts/make_contact_sheets.py [family ...]
"""
import json
import os
import sys
import textwrap

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "pilot", "images")
SHEETS = os.path.join(IMG, "sheets")

CELL = 420
PAD = 16
CAPTION_H = 104          # room for the id line plus a wrapped prompt
FONT_DIR = "/usr/share/fonts/truetype/dejavu"

# Families may define their own condition set (4a2 splits S into S_exp / S_imp), so the
# column order is taken from the manifest and only the ordering is fixed here.
COND_ORDER = ["S", "S_exp", "S_imp", "P", "A"]
# The condition is what the reader is comparing, so give each one a colour.
COND_COLOR = {"S": (176, 0, 32), "S_exp": (176, 0, 32), "S_imp": (204, 85, 0),
              "P": (0, 102, 51), "A": (80, 80, 80)}


def load_fonts():
    try:
        return (ImageFont.truetype(os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf"), 15),
                ImageFont.truetype(os.path.join(FONT_DIR, "DejaVuSans.ttf"), 13))
    except OSError:
        return ImageFont.load_default(), ImageFont.load_default()


def main(families):
    manifest = []
    for fn in sorted(os.listdir(IMG)):
        if fn.startswith("manifest") and fn.endswith(".jsonl"):
            manifest += [json.loads(l) for l in open(os.path.join(IMG, fn)) if l.strip()]
    by_family = {}
    for r in manifest:
        by_family.setdefault(r["family"], {}).setdefault(r["id"], {})[r["condition"]] = r

    os.makedirs(SHEETS, exist_ok=True)
    bold, regular = load_fonts()
    for fam, items in sorted(by_family.items()):
        if families and fam not in families:
            continue
        ids = sorted(items)
        present = {c for conds in items.values() for c in conds}
        CONDS = [c for c in COND_ORDER if c in present]
        # 4 items per sheet keeps each sheet legible at a readable resolution
        for page, start in enumerate(range(0, len(ids), 4)):
            chunk = ids[start:start + 4]
            row_h = CELL + CAPTION_H + PAD
            w = len(CONDS) * CELL + PAD * (len(CONDS) + 1)
            h = len(chunk) * row_h + PAD
            sheet = Image.new("RGB", (w, h), "white")
            draw = ImageDraw.Draw(sheet)
            for r, iid in enumerate(chunk):
                for c, cond in enumerate(CONDS):
                    rec = items[iid].get(cond)
                    x = PAD + c * (CELL + PAD)
                    y = PAD + r * row_h
                    if rec:
                        im = Image.open(os.path.join(ROOT, rec["path"])).convert("RGB")
                        sheet.paste(im.resize((CELL, CELL), Image.LANCZOS), (x, y))
                    draw.rectangle([x, y, x + CELL - 1, y + CELL - 1], outline=(210, 210, 210))

                    ty = y + CELL + 6
                    label = f"{iid}  [{cond}]"
                    draw.text((x, ty), label, fill=COND_COLOR[cond], font=bold)
                    ty += 19
                    prompt = rec["prompt"] if rec else "(not generated)"
                    for line in textwrap.wrap(prompt, width=58)[:4]:
                        draw.text((x, ty), line, fill=(40, 40, 40), font=regular)
                        ty += 16
            out = os.path.join(SHEETS, f"{fam}_p{page+1}.png")
            sheet.save(out)
            print(out)


if __name__ == "__main__":
    main(sys.argv[1:])
