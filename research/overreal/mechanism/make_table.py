"""Print the text-encoder probe results as a table (markdown to stdout, LaTeX
rows to results/text_encoder_table.tex).

Rows: encoder x feature. Columns: per-family accuracy of the pooled probe on
that family's held-out items, then the family-balanced pooled accuracy; the
label-shuffle 95th percentile and the mass-mean accuracy follow.

Usage: python make_table.py
"""
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
FAMILIES = ["cancellation", "attribution", "figurative", "perspectival"]
FAM_TEX = {"cancellation": "Exist.", "attribution": "Mental", "figurative": "Figur.",
           "perspectival": "Persp."}
ENC_NAME = {"clip_l": "CLIP-L/14", "clip_g": "CLIP-bigG/14", "t5": "T5-XXL", "qwen": "Qwen2.5-VL-7B"}
FEAT_NAME = {"span": "target span", "ctrl": "control word", "mean": "prompt mean",
             "pooled": "pooled (FLUX)", "proj": "pooled, projected (SD3)"}
ORDER = ["span", "ctrl", "mean", "pooled", "proj"]


def main():
    res = {}
    for enc in ["clip_l", "clip_g", "t5", "qwen"]:
        p = os.path.join(RES, f"text_encoder_{enc}.json")
        if os.path.exists(p):
            res.update(json.load(open(p)))
    if not res:
        raise SystemExit("no results yet")

    head = f"| encoder | feature | " + " | ".join(FAMILIES) + " | pooled | null95 | mass-mean |"
    print(head)
    print("|" + "---|" * (len(FAMILIES) + 5))
    tex = []
    for enc, r in res.items():
        for feat in ORDER:
            if feat not in r:
                continue
            a = r[feat]["acc"]
            ci = r[feat]["ci"]
            null = r[feat].get("null", {}).get("p95", float("nan"))
            mm = r[feat]["massmean_acc"]["pooled"]
            cells = [f"{a[f]:.2f} [{ci[f][0]:.2f},{ci[f][1]:.2f}]" for f in FAMILIES]
            print(f"| {ENC_NAME[enc]} | {FEAT_NAME[feat]} | " + " | ".join(cells) +
                  f" | {a['pooled']:.2f} | {null:.2f} | {mm:.2f} |")
            tex.append(f"{ENC_NAME[enc]} & {FEAT_NAME[feat]} & " +
                       " & ".join(f"{a[f]:.2f}" for f in FAMILIES) +
                       f" & {a['pooled']:.2f} & {null:.2f} & {mm:.2f} \\\\")
    with open(os.path.join(RES, "text_encoder_table.tex"), "w") as fp:
        fp.write("\n".join(tex) + "\n")
    print("\nLaTeX rows written to results/text_encoder_table.tex")


if __name__ == "__main__":
    main()
