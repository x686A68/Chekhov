"""Layer sweep of the target-span probe: pooled and per-family accuracy per
encoder layer. Reads results/layer_sweep_<encoder>.json, writes
results/layer_sweep.pdf (one panel per encoder, x = layer as a fraction of
depth so the four encoders share an axis).

Usage: python plot_sweep.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
ENC = [("clip_l", "CLIP-L/14"), ("clip_g", "CLIP-bigG/14"), ("t5", "T5-XXL"), ("qwen", "Qwen2.5-VL-7B")]
FAM = [("cancellation", "existence-canceling"), ("attribution", "mental-state"),
       ("figurative", "figurative"), ("perspectival", "perspectival")]
DEPLOYED = {"clip_l": -2, "clip_g": -2, "t5": -1, "qwen": -1}


def main():
    avail = [(e, n) for e, n in ENC if os.path.exists(os.path.join(RES, f"layer_sweep_{e}.json"))]
    fig, axes = plt.subplots(1, len(avail), figsize=(3.2 * len(avail), 2.6), sharey=True)
    if len(avail) == 1:
        axes = [axes]
    for ax, (enc, name) in zip(axes, avail):
        sweep = json.load(open(os.path.join(RES, f"layer_sweep_{enc}.json")))[enc]
        xs = list(range(len(sweep)))
        for fam, label in FAM:
            ax.plot(xs, [a[fam] for a in sweep], lw=1, label=label)
        ax.plot(xs, [a["pooled"] for a in sweep], lw=2, color="k", label="pooled")
        dep = len(sweep) + DEPLOYED[enc]
        ax.axvline(dep, color="gray", ls="--", lw=0.8)
        ax.axhline(0.5, color="gray", ls=":", lw=0.8)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("layer", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("probe accuracy", fontsize=8)
    axes[0].set_ylim(0.4, 1.0)
    axes[-1].legend(fontsize=6, loc="lower right")
    fig.tight_layout()
    out = os.path.join(RES, "layer_sweep.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=150)
    print("saved", out)


if __name__ == "__main__":
    main()
