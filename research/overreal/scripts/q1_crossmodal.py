"""Cross-modal Q1 table: does any entity fail in all six families once both modalities count?

Image labels use direct inspection where it exists (families 2, 3, 4b — the judgements the
VLM judge cannot be trusted on) and the judge elsewhere.

Writes pilot/q1_crossmodal.md.
"""
import json
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXT = os.path.join(ROOT, "pilot", "text")
IMG = os.path.join(ROOT, "pilot", "images")

CORE = ["1_existence", "2_attribution", "3_figurative", "4a_occlusion", "5_use_mention",
        "6_relevance"]
REALIZED_LETTER = {f: {"A"} for f in CORE}


def text_rows():
    rows = []
    for fam in CORE:
        p = os.path.join(TEXT, fam, "results.jsonl")
        if os.path.exists(p):
            rows += [json.loads(l) for l in open(p) if l.strip()]
    return rows


def image_rows():
    human = {}
    ip = os.path.join(IMG, "inspection.jsonl")
    if os.path.exists(ip):
        for l in open(ip):
            if l.strip():
                r = json.loads(l)
                human[(r["id"], r["condition"])] = r["human_letter"]
    rows = []
    for fam in CORE:
        p = os.path.join(IMG, fam, "results.jsonl")
        if not os.path.exists(p):
            continue
        for l in open(p):
            if not l.strip():
                continue
            r = json.loads(l)
            hl = human.get((r["id"], r["condition"]))
            r["realized"] = (hl in REALIZED_LETTER[fam]) if hl else bool(r["judge_realized"])
            r["source"] = "inspection" if hl else "judge"
            rows.append(r)
    return rows


def rate(rows, fam, ent, cond, flag):
    sel = [r for r in rows if r["family"] == fam and r["entity"] == ent
           and r["condition"] == cond]
    return sum(bool(r[flag]) for r in sel) / len(sel) if sel else float("nan")


def main():
    trows, irows = text_rows(), image_rows()
    entities = sorted({r["entity"] for r in trows})

    out = ["# Q1 cross-modal — does any entity fail in all six families?", "",
           "Each cell is `text S / image S`: the suppression-failure rate under the S",
           "condition in each modality. Text pools the three models; images are FLUX.1-dev,",
           "scored by direct inspection for families 2 and 3 and by the VLM judge otherwise.",
           "**bold** marks a cell where the entity failed in at least one modality.", "",
           "| entity | " + " | ".join(f.replace("_", " ") for f in CORE) + " | families failed |",
           "|---|" + "---|" * (len(CORE) + 1)]
    tallies = {}
    for ent in entities:
        cells, failed = [], 0
        for fam in CORE:
            t = rate(trows, fam, ent, "S", "realized_affirmative")
            i = rate(irows, fam, ent, "S", "realized")
            hit = (t > 0) or (i > 0)
            failed += hit
            cell = f"{t:.2f} / {i:.2f}"
            cells.append(f"**{cell}**" if hit else cell)
        tallies[ent] = failed
        out.append(f"| **{ent}** | " + " | ".join(cells) + f" | **{failed}/{len(CORE)}** |")

    out += ["", "## By family, pooled over entities", "",
            "| family | text S | image S | text Δ | image Δ |", "|---|---|---|---|---|"]
    for fam in CORE:
        ts = [r for r in trows if r["family"] == fam]
        isr = [r for r in irows if r["family"] == fam]
        def rr(rows, cond, flag):
            sel = [r for r in rows if r["condition"] == cond]
            return sum(bool(r[flag]) for r in sel) / len(sel) if sel else float("nan")
        ts_S, ts_P = rr(ts, "S", "realized_affirmative"), rr(ts, "P", "realized_affirmative")
        is_S, is_P = rr(isr, "S", "realized"), rr(isr, "P", "realized")
        out.append(f"| {fam.replace('_',' ')} | {ts_S:.2f} | {is_S:.2f} | {ts_P-ts_S:+.2f} | {is_P-is_S:+.2f} |")

    with open(os.path.join(ROOT, "pilot", "q1_crossmodal.md"), "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out[6:]))


if __name__ == "__main__":
    main()
