"""Render an example for the paper: the generated image next to the map of
attention from image tokens to the target tokens (early steps), for one or
more recorded images.

Usage: python example_map.py --model flux --name figurative_prompt_0001__S_s0 [--bins 0 1]
       python example_map.py --model flux --list silent   # list recorded S images with that label
"""
import argparse
import glob
import json
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(ROOT, "..", "..", ".."))
AUTO = os.path.join(REPO, "data", "overreal_v1", "auto", "final__claude-opus-5-api__sample.jsonl")


def labels(model):
    lab = {}
    for line in open(AUTO):
        r = json.loads(line)
        m = re.match(r"(.+)/gen_(.+)_raw_s(\d)$", r["image_id"])
        if m and m.group(2) == model:
            lab[(m.group(1), int(m.group(3)))] = r.get("label")
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="flux")
    ap.add_argument("--name", nargs="*", default=[])
    ap.add_argument("--list", default="")
    ap.add_argument("--bins", nargs="*", type=int, default=[0, 1])
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "examples"))
    args = ap.parse_args()
    d = os.path.join(ROOT, "attn", args.model)

    if args.list:
        lab = labels(args.model)
        for f in sorted(glob.glob(os.path.join(d, "*__S_s*.npz"))):
            z = np.load(f)
            key = (str(z["item_id"]), int(z["seed"]))
            if lab.get(key) == args.list:
                mp = z["map"].astype(np.float32)[args.bins].mean(0)
                print(f"{os.path.basename(f)[:-4]:<45} peak/mean={mp.max()/mp.mean():5.1f}  {str(z['target'])!r:<28} {str(z['prompt'])[:70]}")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    os.makedirs(args.out, exist_ok=True)
    for name in args.name:
        z = np.load(os.path.join(d, name + ".npz"))
        img = Image.open(os.path.join(d, name + ".png")).convert("RGB")
        mp = z["map"].astype(np.float32)[args.bins].mean(0)
        side = int(np.sqrt(mp.size))
        mp = mp.reshape(side, side)
        fig, axes = plt.subplots(1, 2, figsize=(8, 4.1))
        axes[0].imshow(img); axes[0].axis("off")
        axes[1].imshow(img, alpha=0.35)
        axes[1].imshow(mp, cmap="inferno", alpha=0.75, extent=(0, img.width, img.height, 0),
                       interpolation="bilinear")
        axes[1].axis("off")
        fig.suptitle(f"{z['prompt']}\ntarget: {z['target']}", fontsize=8)
        fig.tight_layout()
        out = os.path.join(args.out, name + ".png")
        fig.savefig(out, dpi=150)
        fig.savefig(out.replace(".png", ".pdf"))
        print("saved", out)


if __name__ == "__main__":
    main()
