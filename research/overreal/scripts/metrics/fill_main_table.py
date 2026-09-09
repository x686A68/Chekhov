"""Fill blocks (a) and (b) of the main-results table.

Block (a): mean conventional metrics per generator x family (raw/deployed).
Block (b): outcome rates from the Det gold split; a cell needs >= 10 gold
images, otherwise ---. The best value per column is bolded (min for
D.O./S.O./b, max elsewhere).

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

    # LJ = per-image mean of the two judges; cell mean over images both judged
    lj_by_img = collections.defaultdict(dict)
    for name, fn in (("opus", "lj_subscription"), ("gpt", "lj_gpt")):
        f = DS / "metrics" / f"{fn}.jsonl"
        if f.exists():
            for line in open(f, encoding="utf-8"):
                r = json.loads(line)
                lj_by_img[r["image_id"]][name] = r["score"]
    acc = collections.defaultdict(list)
    for iid, d in lj_by_img.items():
        key = meta.get(iid)
        if key and len(d) == 2:
            acc[key].append((d["opus"] + d["gpt"]) / 2)
    means["lj"] = {k: sum(v) / len(v) for k, v in acc.items() if len(v) >= 10}

    # block (b): outcome rates from the gold split
    import collections as _c
    gold = _c.defaultdict(list)
    for line in open(DS / "det_split.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["split"] == "gold" and r["image_id"] in meta:
            gold[meta[r["image_id"]]].append(set(r["gold_labels"]))
    OUTS = ["disruptive", "silent", "integrated", "withheld"]
    rates = {}
    for key, labels in gold.items():
        n = len(labels)
        if n >= 10:
            rates[key] = {o: sum(o in ls for ls in labels) / n for o in OUTS}

    def build_grid(getval, ncol_types):
        # grid[(row_idx, col_idx)] = float value or None
        grid = {}
        ri = 0
        for label, gen in ROWS:
            if label is None:
                continue
            for fi, fam in enumerate(FAMS):
                ncols = len(ncol_types) + 1
                for ci, ct in enumerate(ncol_types):
                    grid[(ri, fi * ncols + ci)] = getval(gen, fam, ct)
                grid[(ri, fi * ncols + len(ncol_types))] = getval(gen, fam, "_extra")
            ri += 1
        return grid

    def render_block(getval, col_types, fmts, minimize, extra_fmt=None):
        grid = build_grid(getval, col_types)
        ncols = len(col_types) + 1  # + trailing extra column
        best = {}
        for fi in range(len(FAMS)):
            for ci in range(ncols):
                col = fi * ncols + ci
                ct = col_types[ci] if ci < len(col_types) else "_extra"
                vals = [(v, r) for (r, c), v in grid.items() if c == col and v is not None]
                if vals:
                    pick = min(vals) if ct in minimize else max(vals)
                    best[col] = pick[1]
        lines = []
        ri = 0
        for label, gen in ROWS:
            if label is None:
                lines.append("\\midrule")
                continue
            cells = []
            for fi, fam in enumerate(FAMS):
                for ci, ct in enumerate(col_types):
                    col = fi * ncols + ci
                    v = grid.get((ri, col))
                    if v is None:
                        cells.append("---")
                    else:
                        out = fmt_num(fmts[ci], v)
                        if best.get(col) == ri:
                            out = "\\textbf{" + out + "}"
                        cells.append(out)
                ev = grid.get((ri, fi * ncols + len(col_types)))
                cells.append(fmt_num(extra_fmt, ev) if (ev is not None and extra_fmt) else "---")
            lines.append(label + "& " + " & ".join(cells) + " \\\\")
            ri += 1
        return "\n".join(lines)

    block_a = render_block(
        lambda g, f, m: means["lj" if m == "_extra" else m].get((g, f)),
        [m for m, _ in METRICS], [f for _, f in METRICS], minimize=set(),
        extra_fmt="{:.1f}")
    block_b = render_block(
        lambda g, f, o: rates.get((g, f), {}).get(o),
        OUTS, ["{:.2f}"] * 4, minimize={"disruptive", "silent"})

    s = open(TEX, encoding="utf-8").read()
    for name, block in (("(a) Conventional evaluation", block_a),
                        ("(b) Over-realization evaluation", block_b)):
        marker = "\\multicolumn{21}{l}{\\emph{" + name + "}} \\\\\n"
        i = s.index(marker) + len(marker)
        i = s.index("\\midrule", i) + len("\\midrule")
        j = s.index("\\bottomrule" if "(b)" in name else
                    "\\midrule\n\\multicolumn{21}{l}{\\emph{(b)", i)
        s = s[:i] + "\n" + block + "\n" + s[j:]
    open(TEX, "w", encoding="utf-8").write(s)
    print("filled blocks (a) and (b) with bolded column bests")


if __name__ == "__main__":
    main()
