"""Contact sheets for manual inspection: one row per (item, seed), columns =
baseline and conditions. Usage: python intervene_sheets.py --model sd35l --per-family 6 --seed 0
"""
import argparse, collections, json, os, random
from PIL import Image, ImageDraw
ROOT = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(ROOT, "..", "..", ".."))
COLS = ["baseline", "cue_x2", "cue_x4", "cue_x8", "tgt_d8", "rand_x8", "P_baseline", "P_rep_x8"]

def path(model, cond, item, seed):
    fam, pid = item.split("/"); num = pid.split("_")[1]; stem = item.replace("/", "_")
    if cond == "baseline": return os.path.join(REPO, "data", "generation", "images", model, "raw", f"{fam}_{num}__s{seed}.png")
    if cond == "P_baseline": return os.path.join(ROOT, "attn", model, f"{stem}__P_s{seed}.png")
    return os.path.join(ROOT, "intervene", model, cond, f"{stem}__s{seed}.jpg")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--per-family", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="/tmp/claude-2181/-home-jiahao-huang-Chekhov/4f68d830-72f1-42b4-b6e3-3bb9ca26b43f/scratchpad/sheets"); ap.add_argument("--cols", default=",".join(COLS))
    a = ap.parse_args(); cols = a.cols.split(","); os.makedirs(a.out, exist_ok=True)
    pairs = {json.loads(l)["item_id"]: json.loads(l) for l in open(os.path.join(ROOT, "pairs.jsonl"))}
    items = json.load(open(os.path.join(ROOT, "intervene_items.json")))
    have = [i for i in items if all(os.path.exists(path(a.model, c, i, a.seed)) for c in cols if not c.endswith("baseline"))]
    byf = collections.defaultdict(list)
    for i in have: byf[i.split("/")[0]].append(i)
    rng = random.Random(1); W = 220; rows_per = 4; n = 0; legend = []
    for fam in ["cancellation", "attribution", "figurative", "perspectival"]:
        sel = sorted(byf[fam]); rng.shuffle(sel); sel = sel[:a.per_family]
        for b in range(0, len(sel), rows_per):
            chunk = sel[b:b + rows_per]
            sheet = Image.new("RGB", (W * len(cols), (W + 16) * len(chunk) + 16), "white"); d = ImageDraw.Draw(sheet)
            for ci, c in enumerate(cols): d.text((ci * W + 4, 2), c, fill="black")
            for ri, item in enumerate(chunk):
                y = 16 + ri * (W + 16)
                d.text((4, y + 2), f"{item}  target={pairs[item]['target']}", fill="blue")
                for ci, c in enumerate(cols):
                    p = path(a.model, c, item, a.seed)
                    if os.path.exists(p): sheet.paste(Image.open(p).convert("RGB").resize((W, W)), (ci * W, y + 16))
                legend.append(f"{item} | {pairs[item]['s_text']} || P: {pairs[item]['p_text']}")
            n += 1; sheet.save(os.path.join(a.out, f"{a.model}_{fam}_{b // rows_per}.jpg"), quality=85)
    open(os.path.join(a.out, f"{a.model}_legend.txt"), "w").write("\n".join(legend)); print(n, "sheets ->", a.out)

if __name__ == "__main__":
    main()
