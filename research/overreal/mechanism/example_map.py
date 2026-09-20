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
    ap.add_argument("--smooth", type=float, default=1.0, help="gaussian sigma in patches (0 = none)")
    ap.add_argument("--no-title", action="store_true")
    ap.add_argument("--px", type=int, default=512, help="downscale images to this width before embedding")
    ap.add_argument("--pair", default="", help="item stem like figurative_prompt_0084 with --seed: S and P side by side")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quad", default="", help="item stem: original, control, Qwen rewrite, Ideogram rewrite (2 x 4 grid)")
    ap.add_argument("--labels", nargs="*", default=["Original prompt", "Plain-mention control", "Qwen rewrite", "Ideogram rewrite"])
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

    def load_map(name):
        z = np.load(os.path.join(d, name + ".npz"))
        mp = z["map"].astype(np.float32)[args.bins].mean(0)
        side = int(np.sqrt(mp.size))
        mp = mp.reshape(side, side)
        if args.smooth > 0:
            from scipy.ndimage import gaussian_filter
            mp = gaussian_filter(mp, args.smooth)
        im = Image.open(os.path.join(d, name + ".png")).convert("RGB")
        im = im.resize((args.px, int(args.px * im.height / im.width)), Image.LANCZOS)
        return z, im, mp

    if args.quad:
        REPO = os.path.abspath(os.path.join(ROOT, "..", "..", ".."))
        specs = [(d, f"{args.quad}__S_s{args.seed}"), (d, f"{args.quad}__P_s{args.seed}"),
                 (d + "_qwen", f"{args.quad}__X_s{args.seed}"), (d + "_ideogram", f"{args.quad}__X_s{args.seed}")]
        panels = []
        for (dd, name), cond in zip(specs, ["raw", "raw", "qwen", "ideogram"]):
            npz = os.path.join(dd, name + ".npz")
            if os.path.exists(npz):
                z = np.load(npz)
                mp = z["map"].astype(np.float32)[args.bins].mean(0)
                side = int(np.sqrt(mp.size)); mp = mp.reshape(side, side)
                if args.smooth > 0:
                    from scipy.ndimage import gaussian_filter
                    mp = gaussian_filter(mp, args.smooth)
                im = Image.open(os.path.join(dd, name + ".png")).convert("RGB")
                im = im.resize((args.px, int(args.px * im.height / im.width)), Image.LANCZOS)
                panels.append((im, mp))
            else:  # no record (target absent from the rewrite): show the benchmark image only
                stem = args.quad.replace("_prompt_", "_")
                img_path = os.path.join(REPO, "data", "generation", "images", args.model, cond, f"{stem}__s{args.seed}.png")
                im = Image.open(img_path).convert("RGB") if os.path.exists(img_path) else None
                if im is not None:
                    im = im.resize((args.px, int(args.px * im.height / im.width)), Image.LANCZOS)
                panels.append((im, None))
        vmax = max(mp.max() for _, mp in panels if mp is not None)
        fig, axes = plt.subplots(2, 4, figsize=(12, 6.3))
        for i, (img, mp) in enumerate(panels):
            ax_img, ax_map = axes[0, i], axes[1, i]
            if img is not None:
                ax_img.imshow(img)
            ax_img.axis("off"); ax_map.axis("off")
            if mp is not None:
                ax_map.imshow(img, alpha=0.35)
                ax_map.imshow(mp, cmap="inferno", alpha=0.75, vmin=0, vmax=vmax,
                              extent=(0, img.width, img.height, 0), interpolation="bilinear")
            else:
                ax_map.text(0.5, 0.5, "target dropped\nby the rewriter", ha="center", va="center", fontsize=10)
            if not args.no_title:
                ax_img.set_title(args.labels[i], fontsize=9)
        fig.tight_layout(pad=0.3)
        out = os.path.join(args.out, f"{args.quad}_quad_s{args.seed}.png")
        fig.savefig(out, dpi=150); fig.savefig(out.replace(".png", ".pdf"))
        print("saved", out)
        return

    if args.pair:
        panels = [load_map(f"{args.pair}__{side}_s{args.seed}") for side in ("S", "P")]
        vmax = max(mp.max() for _, _, mp in panels)
        fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
        for i, (z, img, mp) in enumerate(panels):
            axes[2 * i].imshow(img); axes[2 * i].axis("off")
            axes[2 * i + 1].imshow(img, alpha=0.35)
            axes[2 * i + 1].imshow(mp, cmap="inferno", alpha=0.75, vmin=0, vmax=vmax,
                                   extent=(0, img.width, img.height, 0), interpolation="bilinear")
            axes[2 * i + 1].axis("off")
            if not args.no_title:
                axes[2 * i].set_title(str(z["prompt"]), fontsize=8)
        fig.tight_layout(pad=0.3)
        out = os.path.join(args.out, f"{args.pair}_pair_s{args.seed}.png")
        fig.savefig(out, dpi=150); fig.savefig(out.replace(".png", ".pdf"))
        print("saved", out)
        return

    for name in args.name:
        z = np.load(os.path.join(d, name + ".npz"))
        img = Image.open(os.path.join(d, name + ".png")).convert("RGB")
        mp = z["map"].astype(np.float32)[args.bins].mean(0)
        side = int(np.sqrt(mp.size))
        mp = mp.reshape(side, side)
        if args.smooth > 0:
            from scipy.ndimage import gaussian_filter
            mp = gaussian_filter(mp, args.smooth)
        fig, axes = plt.subplots(1, 2, figsize=(8, 4.1))
        axes[0].imshow(img); axes[0].axis("off")
        axes[1].imshow(img, alpha=0.35)
        axes[1].imshow(mp, cmap="inferno", alpha=0.75, extent=(0, img.width, img.height, 0),
                       interpolation="bilinear")
        axes[1].axis("off")
        if not args.no_title:
            fig.suptitle(f"{z['prompt']}\ntarget: {z['target']}", fontsize=8)
        fig.tight_layout(pad=0.3)
        out = os.path.join(args.out, name + ".png")
        fig.savefig(out, dpi=150)
        fig.savefig(out.replace(".png", ".pdf"))
        print("saved", out)


if __name__ == "__main__":
    main()
