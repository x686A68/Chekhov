"""Assemble the Q1 tables and the Table 1 candidate cells from the scored text pilot.

Outputs
  pilot/text/q1_entity_table.md   per (entity, family) suppression-failure rate and delta
  pilot/table1_candidates.md      verbatim S-condition generations, per candidate entity,
                                  for the six Table 1 cells
"""
import json
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXT = os.path.join(ROOT, "pilot", "text")

FAMILIES = ["1_existence", "2_attribution", "3_figurative", "4a_occlusion",
            "4b_legibility", "5_use_mention", "6_relevance"]
CORE = [f for f in FAMILIES if f != "4b_legibility"]  # 4b uses the carrier pool


def load():
    rows = []
    for fam in FAMILIES:
        p = os.path.join(TEXT, fam, "results.jsonl")
        if os.path.exists(p):
            rows += [json.loads(l) for l in open(p) if l.strip()]
    return rows


def rate(rows, cond):
    sel = [r for r in rows if r["condition"] == cond]
    return sum(r["realized_affirmative"] for r in sel) / len(sel) if sel else float("nan")


def main():
    rows = load()
    models = sorted({r["model"] for r in rows})
    entities = sorted({r["entity"] for r in rows if r["family"] in CORE})

    # ---- Q1 table -----------------------------------------------------------------
    lines = ["# Q1 — which entity fails in all six families? (text pilot)", "",
             "Cells are `S / P (Δ)` — the suppression-failure rate under S, the licensed",
             "realization rate under P, and licence sensitivity Δ = P − S. Affirmative-realization",
             "scoring. n = 3 items per (entity, family) cell per model.", "",
             "**An entity 'fails' a family when its S rate is above zero — the failure the paper's**",
             "**Table 1 needs is a realization that should not have happened.**", ""]
    for model in models:
        lines += [f"## {model}", "",
                  "| entity | " + " | ".join(f.replace("_", " ") for f in CORE) + " | families failed |",
                  "|---|" + "---|" * (len(CORE) + 1)]
        for ent in entities:
            cells, failed = [], 0
            for fam in CORE:
                sel = [r for r in rows if r["model"] == model and r["family"] == fam and r["entity"] == ent]
                s, p = rate(sel, "S"), rate(sel, "P")
                if s > 0:
                    failed += 1
                cells.append(f"{s:.2f} / {p:.2f} ({p-s:+.2f})")
            lines.append(f"| **{ent}** | " + " | ".join(cells) + f" | **{failed}/{len(CORE)}** |")
        lines.append("")

    # pooled over models
    lines += ["## pooled over all models", "",
              "| entity | " + " | ".join(f.replace("_", " ") for f in CORE) + " | families failed |",
              "|---|" + "---|" * (len(CORE) + 1)]
    for ent in entities:
        cells, failed = [], 0
        for fam in CORE:
            sel = [r for r in rows if r["family"] == fam and r["entity"] == ent]
            s, p = rate(sel, "S"), rate(sel, "P")
            if s > 0:
                failed += 1
            cells.append(f"{s:.2f} / {p:.2f} ({p-s:+.2f})")
        lines.append(f"| **{ent}** | " + " | ".join(cells) + f" | **{failed}/{len(CORE)}** |")
    lines.append("")

    with open(os.path.join(TEXT, "q1_entity_table.md"), "w") as f:
        f.write("\n".join(lines))

    # ---- Table 1 candidate cells ---------------------------------------------------
    out = ["# Table 1 candidates — verbatim generations", "",
           "One entity, six failures. Every cell below is a **verbatim** model generation",
           "under the S condition, never an invented example (DECISIONS.md #13). A cell is",
           "empty when no model over-realized for that (entity, family) pair — that is the",
           "evidence that the entity does *not* fail in all six families.", ""]
    by_cell = defaultdict(list)
    for r in rows:
        if r["condition"] == "S" and r["realized_affirmative"] and r["family"] in CORE:
            by_cell[(r["entity"], r["family"])].append(r)

    for ent in entities:
        n_failed = sum(1 for fam in CORE if by_cell[(ent, fam)])
        out += [f"## {ent} — fails {n_failed}/{len(CORE)} families", ""]
        for fam in CORE:
            hits = by_cell[(ent, fam)]
            out.append(f"### {fam.replace('_', ' ')}")
            if not hits:
                out += ["", "_no over-realization observed in any model_", ""]
                continue
            h = hits[0]
            prompt_text = " || ".join(t["content"] for t in h["prompt"])
            out += ["",
                    f"- **prompt (S)**: {prompt_text}",
                    f"- **model**: `{h['model']}`",
                    f"- **generation**: {h['output'].strip()}",
                    f"- _{len(hits)} of the model x item S generations over-realized for this cell_",
                    ""]
        out.append("")

    # image-modality cells, so a cross-modal Table 1 can be assembled
    img_dir = os.path.join(ROOT, "pilot", "images")
    human = {}
    ip = os.path.join(img_dir, "inspection.jsonl")
    if os.path.exists(ip):
        for l in open(ip):
            if l.strip():
                r = json.loads(l)
                human[(r["id"], r["condition"])] = r
    img_by_cell = defaultdict(list)
    for fam in CORE:
        p = os.path.join(img_dir, fam, "results.jsonl")
        if not os.path.exists(p):
            continue
        for l in open(p):
            if not l.strip():
                continue
            r = json.loads(l)
            if r["condition"] != "S":
                continue
            h = human.get((r["id"], "S"))
            realized = (h["human_letter"] == "A") if h else bool(r["judge_realized"])
            if realized:
                r["_note"] = h.get("note", "") if h else ""
                r["_source"] = "direct inspection" if h else "VLM judge"
                img_by_cell[(r["entity"], fam)].append(r)

    out += ["", "---", "", "# Image-modality cells (FLUX.1-dev)", "",
            "Same rule: every entry is a real generation. The image path is relative to",
            "`research/overreal/`.", ""]
    for ent in entities:
        n = sum(1 for fam in CORE if img_by_cell[(ent, fam)])
        out += [f"## {ent} — fails {n}/{len(CORE)} families in images", ""]
        for fam in CORE:
            hits = img_by_cell[(ent, fam)]
            out.append(f"### {fam.replace('_', ' ')}")
            if not hits:
                out += ["", "_no over-realization observed_", ""]
                continue
            h = hits[0]
            out += ["",
                    f"- **prompt (S)**: {h['prompt']}",
                    f"- **image**: `{h['path']}`",
                    f"- **scored by**: {h['_source']}"
                    + (f" — {h['_note']}" if h.get("_note") else ""),
                    f"- _{len(hits)} of the {fam} S images over-realized for this entity_",
                    ""]
        out.append("")

    with open(os.path.join(ROOT, "pilot", "table1_candidates.md"), "w") as f:
        f.write("\n".join(out))

    print("wrote pilot/text/q1_entity_table.md and pilot/table1_candidates.md")
    for ent in entities:
        n = sum(1 for fam in CORE if by_cell[(ent, fam)])
        print(f"  {ent:<10} fails {n}/{len(CORE)} core families")


if __name__ == "__main__":
    main()
