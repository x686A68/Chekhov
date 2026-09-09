"""Fill block (a) of the main-results table with computed metric means.

Cells: mean metric over a generator's raw/deployed images of one family
(the population matching block (b)'s outcome rows). CS/PS/HP come from the
full runs; VQ from the eval sample when present; LJ stays --- until the
judge exists. Formats: CS 0.00, VQ 0.00, PS 0.0, HP 0.000.
"""
import json
import collections
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DS = ROOT / "data" / "overreal_v1"
TEX = ROOT / "Chekhov_paper_ICLR" / "overrealization.tex"

ROWS = [("GPT-Image    ", "gpt-image-1.5"),
        ("Nano Banana  ", "gemini-2.5-flash-image-api"),
        ("Ideogram     ", "ideogram-v3"),
        (None, None),
        ("FLUX.1-dev   ", "flux.1-dev"),
        ("Qwen-Image   ", "qwen-image"),
        ("OmniGen2     ", "omnigen2"),
        ("SD3.5-Medium ", "sd3.5-medium"),
        ("SD3.5-Large  ", "sd3.5-large")]
FAMS = ["cancellation", "attribution", "figurative", "perspectival"]
METRICS = [("clipscore", "{:.2f}"), ("vqascore", "{:.2f}"),
           ("pickscore", "{:.1f}"), ("hpsv2", "{:.2f}")]


def fmt_num(fmt, v):
    out = fmt.format(v)
    return out[1:] if out.startswith("0.") else out


def main():
    meta = {}
    for line in open(DS / "metadata.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if (r.get("file_name") and not r.get("refused")
                and r.get("source_folder") is None
                and r.get("prompt_cond") in ("raw", "deployed")):
            meta[r["image_id"]] = (r["generator"], r["family"])

    means = {}
    for metric, _ in METRICS:
        p = DS / "metrics" / f"{metric}.jsonl"
        acc = collections.defaultdict(list)
        if p.exists():
            for line in open(p, encoding="utf-8"):
                r = json.loads(line)
                key = meta.get(r["image_id"])
                if key:
                    acc[key].append(r["score"])
        means[metric] = {k: sum(v) / len(v) for k, v in acc.items()}

    lines = []
    for label, gen in ROWS:
        if label is None:
            lines.append("\\midrule")
            continue
        cells = []
        for fam in FAMS:
            for metric, fmt in METRICS:
                v = means[metric].get((gen, fam))
                cells.append(fmt_num(fmt, v) if v is not None else "---")
            cells.append("---")   # LJ
        lines.append(label + "& " + " & ".join(cells) + " \\\\")
    block = "\n".join(lines)

    s = open(TEX, encoding="utf-8").read()
    marker = "\\multicolumn{21}{l}{\\emph{(a) Conventional evaluation}} \\\\\n"
    i = s.index(marker) + len(marker)
    i = s.index("\\midrule", i) + len("\\midrule")   # after the CS/VQ/... subheader
    j = s.index("\\midrule\n\\multicolumn{21}{l}{\\emph{(b)", i)
    s = s[:i] + "\n" + block + "\n" + s[j:]
    open(TEX, "w", encoding="utf-8").write(s)
    n = sum(1 for m, _ in METRICS for _ in means[m])
    print(f"filled block (a); populated cells from {', '.join(m for m,_ in METRICS if means[m])}")


if __name__ == "__main__":
    main()
