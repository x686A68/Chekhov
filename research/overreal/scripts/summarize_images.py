"""Per-family realization rates and licence sensitivity for the image pilot.

Uses the VLM judge verdicts. Where direct inspection exists (families 2 and 4b) the
same statistics are recomputed from the human labels, so the report can say how much
the headline number moves when the judge is replaced by inspection.

Writes pilot/images/summary.json.
"""
import json
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "pilot", "images")

# the letter that counts as "realized" per family, for human-labelled rows
REALIZED_LETTER = {
    "1_existence": {"A"}, "2_attribution": {"A"}, "3_figurative": {"A"},
    "4a_occlusion": {"A"}, "4b_legibility": {"A"}, "5_use_mention": {"A"},
    "6_relevance": {"A"},
}


def rates(rows, flag):
    out = {}
    for cond in "SPA":
        sel = [r for r in rows if r["condition"] == cond and r.get(flag) is not None]
        out[cond] = round(sum(bool(r[flag]) for r in sel) / len(sel), 4) if sel else None
        out[f"n_{cond}"] = len(sel)
    if out["S"] is not None and out["P"] is not None:
        out["delta"] = round(out["P"] - out["S"], 4)
    return out


def main():
    judged = []
    for fam in sorted(os.listdir(IMG)):
        p = os.path.join(IMG, fam, "results.jsonl")
        if os.path.isfile(p):
            judged += [json.loads(l) for l in open(p) if l.strip()]
    if not judged:
        print("no judged images yet")
        return

    human = {}
    ip = os.path.join(IMG, "inspection.jsonl")
    if os.path.exists(ip):
        for l in open(ip):
            if l.strip():
                r = json.loads(l)
                human[(r["id"], r["condition"])] = r["human_letter"]

    by_family = defaultdict(list)
    for r in judged:
        by_family[r["family"]].append(r)

    summary = {}
    for fam, rows in sorted(by_family.items()):
        entry = {"n_images": len(rows), "judge": rates(rows, "judge_realized")}
        # per-entity, for the Q1 cross-check in the image modality
        per_ent = {}
        for ent in sorted({r["entity"] for r in rows}):
            per_ent[ent] = rates([r for r in rows if r["entity"] == ent], "judge_realized")
        entry["judge_by_entity"] = per_ent
        if fam == "5_use_mention":
            entry["judge_referent_present"] = rates(rows, "judge2_referent_present")
        if fam == "4b_legibility":
            entry["judge_reader_can_see"] = rates(rows, "judge2_reader_can_see")

        hrows = []
        for r in rows:
            hl = human.get((r["id"], r["condition"]))
            if hl is not None:
                hrows.append({**r, "human_realized": hl in REALIZED_LETTER[fam]})
        if hrows:
            entry["inspection"] = rates(hrows, "human_realized")
        summary[fam] = entry

    with open(os.path.join(IMG, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"{'family':<16}{'S':>8}{'P':>8}{'A':>8}{'delta':>8}   (judge)")
    for fam, e in summary.items():
        j = e["judge"]
        fmt = lambda v: f"{v:>8.2f}" if isinstance(v, float) else f"{'--':>8}"
        print(f"{fam:<16}{fmt(j['S'])}{fmt(j['P'])}{fmt(j['A'])}{fmt(j.get('delta'))}")
        if "inspection" in e:
            i = e["inspection"]
            print(f"{'  (inspected)':<16}{fmt(i['S'])}{fmt(i['P'])}{fmt(i['A'])}{fmt(i.get('delta'))}")


if __name__ == "__main__":
    main()
