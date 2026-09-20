"""Figures for 6.3: the distribution of the target's relative attention r
(per-token share of the target divided by the prompt's average per-token
share; 1 = an ordinary word) by outcome label, for original and rewritten
prompts, and the distribution of r for original versus rewritten prompts.

Reads results/attn_by_label.json (written by the collection snippet) and
results/attn_relative.json is not needed. Writes results/attn_by_label.pdf
and results/attn_orig_vs_rewrite.pdf (+ png).

Usage: python plot_by_label.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sys
MEASURE = sys.argv[1] if len(sys.argv) > 1 else "r"   # "r" (length-corrected) or "a" (per-token share, %)
YLAB = {"r": "relative attention $r$ of the target", "a": "attention share $a_\\mathrm{t}$ of the target (%)"}[MEASURE]

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
MODELS = [("flux", "FLUX.1-dev"), ("sd35l", "SD3.5-Large"), ("qwen-image", "Qwen-Image")]
LABELS = [("disruptive", "Disruptive"), ("silent", "Silent"), ("integrated", "Integrated"), ("withheld", "Withheld")]
C_ORIG, C_REW = "#4C72B0", "#DD8452"


def main():
    rows = json.load(open(os.path.join(RES, "attn_by_label.json")))

    # ---- figure 1: r by outcome label, original vs rewrite ------------------
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0), sharey=True)
    for ax, (m, name) in zip(axes, MODELS):
        pos = np.arange(len(LABELS))
        for off, conds, color, lab in [(-0.18, ("raw",), C_ORIG, "original prompt"),
                                       (0.18, ("qwen", "ideogram"), C_REW, "rewritten prompt")]:
            data = [[x[MEASURE] for x in rows if x["model"] == m and x["cond"] in conds and x["label"] == L]
                    for L, _ in LABELS]
            parts = ax.violinplot(data, positions=pos + off, widths=0.32, showmedians=True, showextrema=False)
            for b in parts["bodies"]:
                b.set_facecolor(color); b.set_edgecolor(color); b.set_alpha(0.55)
            parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(1.2)
            for p, d in zip(pos + off, data):
                ax.text(p, -0.12 if MEASURE == "r" else -0.4, str(len(d)), ha="center", va="top", fontsize=6, color=color)
        if MEASURE == "r":
            ax.axhline(1.0, color="gray", ls=":", lw=0.8)
        ax.set_xticks(pos); ax.set_xticklabels([l for _, l in LABELS], fontsize=8)
        ax.set_title(name, fontsize=9); ax.tick_params(labelsize=7)
        ax.set_ylim(-0.3 if MEASURE == "r" else -1.0, 3.5 if MEASURE == "r" else 12)
    axes[0].set_ylabel(YLAB, fontsize=8)
    h = [plt.Rectangle((0, 0), 1, 1, color=C_ORIG, alpha=0.55), plt.Rectangle((0, 0), 1, 1, color=C_REW, alpha=0.55)]
    axes[-1].legend(h, ["original prompt", "rewritten prompt"], fontsize=7, loc="upper right")
    fig.tight_layout()
    out = os.path.join(RES, f"attn_by_label_{MEASURE}.pdf")
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=150)
    print("saved", out)

    # ---- figure 2: distribution of the measure, original vs rewrite, with the
    # over-realization rate (disruptive + silent over all four labels) per bin
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.8), sharey=True)
    bins = np.linspace(0, 3.5, 36) if MEASURE == "r" else np.linspace(0, 12, 36)
    rbins = np.linspace(0, 3.5, 11) if MEASURE == "r" else np.linspace(0, 12, 13)
    MIN_N = 12
    for ax, (m, name) in zip(axes, MODELS):
        ax2 = ax.twinx()
        for conds, color, lab in [(("raw",), C_ORIG, "original"), (("qwen", "ideogram"), C_REW, "rewritten")]:
            sub = [x for x in rows if x["model"] == m and x["cond"] in conds]
            d = [x[MEASURE] for x in sub]
            ax.hist(d, bins=bins, density=True, alpha=0.4, color=color, label=f"{lab} (median {np.median(d):.2f})")
            xs, ys = [], []
            for lo, hi in zip(rbins[:-1], rbins[1:]):
                inb = [x for x in sub if lo <= x[MEASURE] < hi]
                if len(inb) >= MIN_N:
                    xs.append(0.5 * (lo + hi))
                    ys.append(np.mean([x["label"] in ("disruptive", "silent") for x in inb]))
            ax2.plot(xs, ys, marker="o", ms=3, lw=1.3, color=color, label=f"{lab}: over-realization rate")
        ax2.set_ylim(0, 1.0)
        ax2.tick_params(labelsize=7)
        if ax is axes[-1]:
            ax2.set_ylabel("over-realization rate in bin", fontsize=8)
        else:
            ax2.set_yticklabels([])
        if MEASURE == "r":
            ax.axvline(1.0, color="gray", ls=":", lw=0.8)
        ax.set_title(name, fontsize=9); ax.tick_params(labelsize=7)
        ax.set_xlabel(YLAB, fontsize=8)
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=5.5, loc="upper right")
    axes[0].set_ylabel("density", fontsize=8)
    fig.tight_layout()
    out = os.path.join(RES, f"attn_orig_vs_rewrite_{MEASURE}.pdf")
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=150)
    print("saved", out)


FAMS = [("cancellation", "Existence-canceling"), ("attribution", "Mental-state"),
        ("figurative", "Figurative"), ("perspectival", "Perspectival")]


def grid():
    """3 rows (models) x 4 columns (families): distribution of the measure for
    original and rewritten prompts, with the per-bin over-realization rate."""
    rows = json.load(open(os.path.join(RES, "attn_by_label.json")))
    bins = np.linspace(0, 3.5, 29) if MEASURE == "r" else np.linspace(0, 12, 29)
    rbins = np.linspace(0, 3.5, 8) if MEASURE == "r" else np.linspace(0, 12, 9)
    MIN_N = 10
    fig, axes = plt.subplots(3, 4, figsize=(11, 6.6), sharex=True)
    for i, (m, mname) in enumerate(MODELS):
        for j, (f, fname) in enumerate(FAMS):
            ax = axes[i, j]; ax2 = ax.twinx()
            for conds, color, lab in [(("raw",), C_ORIG, "original"), (("qwen", "ideogram"), C_REW, "rewritten")]:
                sub = [x for x in rows if x["model"] == m and x["cond"] in conds and x["family"] == f]
                d = [x[MEASURE] for x in sub]
                if d:
                    ax.hist(d, bins=bins, density=True, alpha=0.4, color=color, label=f"{lab} (n={len(d)})")
                xs, ys = [], []
                for lo, hi in zip(rbins[:-1], rbins[1:]):
                    inb = [x for x in sub if lo <= x[MEASURE] < hi]
                    if len(inb) >= MIN_N:
                        xs.append(0.5 * (lo + hi))
                        ys.append(np.mean([x["label"] in ("disruptive", "silent") for x in inb]))
                ax2.plot(xs, ys, marker="o", ms=2.5, lw=1.2, color=color)
            ax2.set_ylim(0, 1.0); ax2.tick_params(labelsize=6)
            if j < 3:
                ax2.set_yticklabels([])
            else:
                ax2.set_ylabel("over-realization rate", fontsize=7)
            if MEASURE == "r":
                ax.axvline(1.0, color="gray", ls=":", lw=0.8)
            ax.tick_params(labelsize=6)
            if i == 0:
                ax.set_title(fname, fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{mname}\ndensity", fontsize=7)
            if i == 2:
                ax.set_xlabel(YLAB, fontsize=7)
            ax.legend(fontsize=5, loc="upper right")
    fig.tight_layout()
    out = os.path.join(RES, f"attn_grid_{MEASURE}.pdf")
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=150)
    print("saved", out)


def target_vs_cue():
    """Original prompts only: distribution of the target's and of the cue
    words' per-token share, each with the per-bin over-realization rate."""
    rows = [x for x in json.load(open(os.path.join(RES, "attn_by_label.json"))) if x["cond"] == "raw" and x["a_c"] is not None]
    key_t, key_c = ("a", "a_c") if MEASURE == "a" else ("r", "r_c")
    bins = np.linspace(0, 12, 37) if MEASURE == "a" else np.linspace(0, 3.5, 36)
    rbins = np.linspace(0, 12, 13) if MEASURE == "a" else np.linspace(0, 3.5, 11)
    C_T, C_C = "#55A868", "#C44E52"
    MIN_N = 12
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1), sharey=True)
    handles, labels = [], []
    for ax, (m, name) in zip(axes, MODELS):
        ax2 = ax.twinx()
        sub = [x for x in rows if x["model"] == m]
        for key, color, lab in [(key_t, C_T, "target words"), (key_c, C_C, "cue words")]:
            d = [x[key] for x in sub]
            ax.hist(d, bins=bins, density=True, alpha=0.4, color=color, label=f"{lab}: distribution")
            xs, ys = [], []
            for lo, hi in zip(rbins[:-1], rbins[1:]):
                inb = [x for x in sub if lo <= x[key] < hi]
                if len(inb) >= MIN_N:
                    xs.append(0.5 * (lo + hi)); ys.append(np.mean([x["label"] in ("disruptive", "silent") for x in inb]))
            ax2.plot(xs, ys, marker="o", ms=3, lw=1.3, color=color, label=f"{lab}: OR rate")
        ax2.set_ylim(0, 1.0); ax2.tick_params(labelsize=7)
        if ax is axes[-1]:
            ax2.set_ylabel("OR rate", fontsize=8)
        else:
            ax2.set_yticklabels([])
        ax.set_title(name, fontsize=9); ax.tick_params(labelsize=7)
        ax.set_xlabel("attention share per token (%)" if MEASURE == "a" else "relative attention", fontsize=8)
        if ax is axes[0]:
            h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
            handles, labels = h1 + h2, l1 + l2
    axes[0].set_ylabel("density", fontsize=8)
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out = os.path.join(RES, f"attn_target_vs_cue_{MEASURE}.pdf")
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=150)
    print("saved", out)


if __name__ == "__main__":
    if "cue" in sys.argv:
        target_vs_cue()
    elif "grid" in sys.argv:
        grid()
    else:
        main()
