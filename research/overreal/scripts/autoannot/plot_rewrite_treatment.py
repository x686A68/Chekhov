"""Figure for 6.2: how the rewriters treat the target, and the over-realization
rate by treatment.

Inputs: data/generation/rewrite_target_treatment.jsonl (one row per rewritten
prompt: category 1 dropped, 2 explicitly excluded, 3 mentioned, 4 explicitly
to be depicted; rule-based first pass, then read and corrected by hand),
data/generation/rewrite_treatment_excluded_items.txt (existence-canceling
items whose target is a normal part of the scene, left out), the eval sample
and the Opus labels of the rewritten-prompt images.

Output: Chekhov_paper_ICLR/figures/rewrite_treatment.pdf (and .png here).
"""
import collections
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import derive_label, load_meta

ROOT = Path(__file__).resolve().parents[4]
GEN = ROOT / "data" / "generation"
COL = ["#86b6ef", "#1c5cab", "#f4a582", "#b2182b"]   # cool: target absent from the prompt; warm: present
TXT = ["#0d366b", "white", "#5a1a0e", "white"]
CAT = ["dropped", "explicitly excluded", "mentioned", "explicitly depicted"]
FAM = [("cancellation", "Existence-\ncanceling"), ("attribution", "Mental-state"),
       ("figurative", "Figurative"), ("perspectival", "Perspectival")]


def main():
    excl = set((GEN / "rewrite_treatment_excluded_items.txt").read_text().split())
    rows = [json.loads(l) for l in open(GEN / "rewrite_target_treatment.jsonl")]
    rows = [r for r in rows if r["item_id"] not in excl]
    cat = {(r["item_id"], r["expander"]): r["category"] for r in rows}
    c = collections.Counter((r["family"], r["expander"], r["category"]) for r in rows)
    dist = {}
    for fam, _ in FAM:
        for ex in ("qwen", "ideogram"):
            n = sum(c[(fam, ex, k)] for k in (1, 2, 3, 4))
            dist[(fam, ex)] = [c[(fam, ex, k)] / n for k in (1, 2, 3, 4)]
    meta = load_meta()
    sample = {json.loads(l)["image_id"] for l in open(ROOT / "data" / "overreal_v1" / "eval_sample.jsonl")}
    o = collections.defaultdict(lambda: [0, 0])
    for l in open(ROOT / "data" / "overreal_v1" / "auto" / "final__claude-opus-5-api__sample.jsonl"):
        r = json.loads(l)
        m = meta.get(r["image_id"])
        if r["image_id"] not in sample or m.get("prompt_cond") not in ("qwen", "ideogram") or m["item_id"] in excl:
            continue
        lab = derive_label(r["answers"], r["questions"])
        if lab not in ("disruptive", "silent", "integrated", "withheld"):
            continue
        k = cat[(m["item_id"], m["prompt_cond"])]
        o[k][1] += 1
        o[k][0] += lab in ("disruptive", "silent")
    orate = [o[k][0] / o[k][1] for k in (1, 2, 3, 4)]

    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.edgecolor": "#888", "axes.linewidth": 0.6})
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.45), gridspec_kw={"width_ratios": [2.3, 1]})
    labels, ys, y = [], [], 0
    for fam, fn in FAM:
        y0 = y
        for ex, exn in (("qwen", "Qwen"), ("ideogram", "Ideogram")):
            left = 0
            for k, (val, col) in enumerate(zip(dist[(fam, ex)], COL)):
                ax.barh(y, val, left=left, color=col, height=0.72, edgecolor="white", linewidth=1)
                if val >= 0.12:
                    ax.text(left + val / 2, y, f"{100*val:.0f}", ha="center", va="center", color=TXT[k], fontsize=7)
                left += val
            labels.append(exn)
            ys.append(y)
            y += 1
        ax.text(-0.27, (y0 + y - 1) / 2, fn, ha="right", va="center", fontsize=8, fontweight="bold",
                transform=ax.get_yaxis_transform(), linespacing=1.0)
        y += 0.45
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(["0", "25", "50", "75", "100%"])
    ax.set_xlabel("proportion of rewritten prompts")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in COL], labels=CAT, loc="upper center",
              bbox_to_anchor=(0.3, 1.25), ncol=4, frameon=False, fontsize=7, handlelength=1,
              columnspacing=0.9, handletextpad=0.4)
    ax.text(-0.42, 1.16, "(a)", transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
    for k, r in enumerate(orate):
        ax2.bar(k, r, color=COL[k], width=0.7)
        ax2.text(k, r + 0.02, f"{100*r:.0f}%", ha="center", va="bottom", fontsize=7)
    ax2.set_ylim(0, 1)
    ax2.set_yticks([0, 0.5, 1])
    ax2.set_yticklabels(["0", "50", "100%"])
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(["dropped", "excluded", "mentioned", "depicted"], rotation=35, ha="right")
    ax2.set_ylabel("over-realization rate", labelpad=4)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.tick_params(length=2)
    ax2.text(-0.55, 1.16, "(b)", transform=ax2.transAxes, fontsize=9, fontweight="bold", va="top")
    fig.tight_layout()
    fig.subplots_adjust(top=0.84, wspace=0.5, left=0.235)
    fig.savefig(ROOT / "Chekhov_paper_ICLR" / "figures" / "rewrite_treatment.pdf")
    fig.savefig(Path(__file__).with_name("rewrite_treatment.png"), dpi=200)
    print("saved; over-realization by treatment:", [round(x, 3) for x in orate])


if __name__ == "__main__":
    main()
