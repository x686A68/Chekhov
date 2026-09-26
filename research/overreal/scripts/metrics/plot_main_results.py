"""Main-results figure: over-realization rate per generator and family, DO/SO stacked.

Data: the Opus judge run on the eval sample (Table 1(b) population), same filters as
fill_main_table.py --outcomes auto. Writes figures/main_results.{pdf,png}.
"""
import collections
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "research" / "overreal" / "scripts" / "autoannot"))
from common import derive_label  # noqa: E402

DS = ROOT / "data" / "overreal_v1"
OUT = ROOT / "Chekhov_paper_ICLR" / "figures" / "main_results"
FAMS = [("cancellation", "Existence-canceling"), ("attribution", "Mental-state"),
        ("figurative", "Figurative"), ("perspectival", "Perspectival")]
GENS = [("gpt-image-1.5", "GPT"), ("gemini-2.5-flash-image-api", "NB"),
        ("ideogram-v3", "Ideo"), ("flux.1-dev", "FLUX"), ("qwen-image", "Qwen"),
        ("omnigen2", "Omni"), ("sd3.5-medium", "SD-M"), ("sd3.5-large", "SD-L")]
PROP = {"gpt-image-1.5", "gemini-2.5-flash-image-api", "ideogram-v3"}


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def rates():
    meta = {}
    for r in read_jsonl(DS / "metadata.jsonl"):
        if (r.get("file_name") and not r.get("refused") and r.get("source_folder") is None
                and r.get("prompt_cond") in ("raw", "deployed")):
            meta[r["image_id"]] = (r["generator"], r["family"])
    sample = {r["image_id"] for r in read_jsonl(DS / "eval_sample.jsonl")}
    cells = collections.defaultdict(list)
    for r in read_jsonl(DS / "auto" / "final__claude-opus-5-api__sample.jsonl"):
        if r["image_id"] not in sample or r["image_id"] not in meta:
            continue
        lab = derive_label(r["answers"], r["questions"])
        if lab:
            cells[meta[r["image_id"]]].append(lab)
    out = {}
    for k, labs in cells.items():
        n = len(labs)
        out[k] = (sum(l == "disruptive" for l in labs) / n, sum(l == "silent" for l in labs) / n, n)
    return out


def main():
    R = rates()
    plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.linewidth": 0.5,
                         "xtick.major.width": 0.5, "ytick.major.width": 0.5})
    fig, axes = plt.subplots(1, 4, figsize=(6.9, 1.6), sharey=True)
    c_prop, c_open = "#b2412f", "#2f5f8f"          # proprietary red, open blue
    for ax, (fam, title) in zip(axes, FAMS):
        for i, (g, name) in enumerate(GENS):
            do, so, n = R.get((g, fam), (0, 0, 0))
            base = c_prop if g in PROP else c_open
            ax.bar(i, do, color=base, width=0.72, linewidth=0)
            ax.bar(i, so, bottom=do, color=base, alpha=0.4, width=0.72, linewidth=0)
            ax.text(i, do + so + 0.02, f"{do + so:.2f}"[1:], ha="center", va="bottom", fontsize=5.5)
        ax.set_title(title, fontsize=7.5, pad=3)
        ax.set_xticks(range(len(GENS)))
        ax.set_xticklabels([n for _, n in GENS], rotation=45, ha="right", rotation_mode="anchor", fontsize=6)
        ax.set_ylim(0, 1.08)
        ax.axvline(2.5, color="0.7", linewidth=0.5, linestyle=":")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(length=2)
    axes[0].set_ylabel("share of images")
    handles = [Patch(color=c_prop, label="proprietary, disruptive"),
               Patch(color=c_prop, alpha=0.4, label="proprietary, silent"),
               Patch(color=c_open, label="open, disruptive"),
               Patch(color=c_open, alpha=0.4, label="open, silent")]
    fig.legend(handles=handles, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.02),
               frameon=False, fontsize=6, handlelength=1.2, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print("wrote", OUT.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
