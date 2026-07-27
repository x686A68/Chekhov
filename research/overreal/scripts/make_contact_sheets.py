"""Build contact sheets for direct inspection.

Reading 250 full-resolution PNGs one at a time is not practical, so images are tiled
into labelled sheets: one row per item, one column per condition, each cell captioned
with the item id and condition. The inspection verdicts are recorded against those ids.

Usage: python scripts/make_contact_sheets.py [family ...]
"""
import json
import os
import sys

from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "pilot", "images")
SHEETS = os.path.join(IMG, "sheets")

CELL = 384
PAD = 22
CONDS = ["S", "P", "A"]


def main(families):
    manifest = []
    for fn in sorted(os.listdir(IMG)):
        if fn.startswith("manifest") and fn.endswith(".jsonl"):
            manifest += [json.loads(l) for l in open(os.path.join(IMG, fn)) if l.strip()]
    by_family = {}
    for r in manifest:
        by_family.setdefault(r["family"], {}).setdefault(r["id"], {})[r["condition"]] = r

    os.makedirs(SHEETS, exist_ok=True)
    for fam, items in sorted(by_family.items()):
        if families and fam not in families:
            continue
        ids = sorted(items)
        # 4 items per sheet keeps each sheet legible at a readable resolution
        for page, start in enumerate(range(0, len(ids), 4)):
            chunk = ids[start:start + 4]
            w = len(CONDS) * CELL + PAD * (len(CONDS) + 1)
            h = len(chunk) * (CELL + PAD) + PAD
            sheet = Image.new("RGB", (w, h), "white")
            draw = ImageDraw.Draw(sheet)
            for r, iid in enumerate(chunk):
                for c, cond in enumerate(CONDS):
                    rec = items[iid].get(cond)
                    x = PAD + c * (CELL + PAD)
                    y = PAD + r * (CELL + PAD)
                    if rec:
                        im = Image.open(os.path.join(ROOT, rec["path"])).convert("RGB")
                        sheet.paste(im.resize((CELL, CELL), Image.LANCZOS), (x, y))
                    draw.text((x, y - 15), f"{iid} [{cond}]", fill="black")
            out = os.path.join(SHEETS, f"{fam}_p{page+1}.png")
            sheet.save(out)
            print(out)


if __name__ == "__main__":
    main(sys.argv[1:])
