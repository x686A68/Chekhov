"""Phase 3 — agreement between the VLM judge and direct inspection.

Direct-inspection verdicts live in pilot/images/inspection.jsonl, one line per image:

    {"id": "...", "condition": "S", "family": "2_attribution", "human_letter": "A",
     "note": "optional free text"}

They are produced by looking at every image of families 2 and 4b (GOAL.md Phase 3) and
recording the letter that is visibly the case, using the same option set the judge saw.

Writes pilot/judge_agreement.json: per-family raw agreement, Cohen's kappa, and the
disagreeing image list so the failures can be read one by one.
"""
import json
import os
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "pilot", "images")


def kappa(a, b):
    """Cohen's kappa for two equal-length label sequences."""
    n = len(a)
    if n == 0:
        return None
    labels = sorted(set(a) | set(b))
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    if pe == 1.0:
        return None  # undefined: both raters constant on the same label
    return (po - pe) / (1 - pe)


def main():
    insp_path = os.path.join(IMG, "inspection.jsonl")
    if not os.path.exists(insp_path):
        print("no inspection.jsonl yet")
        return
    human = {}
    for l in open(insp_path):
        if l.strip():
            r = json.loads(l)
            human[(r["id"], r["condition"])] = r

    judged = {}
    for fam in os.listdir(IMG):
        p = os.path.join(IMG, fam, "results.jsonl")
        if os.path.isfile(p):
            for l in open(p):
                if l.strip():
                    r = json.loads(l)
                    judged[(r["id"], r["condition"])] = r

    per_family = defaultdict(lambda: {"n": 0, "agree": 0, "judge": [], "human": [],
                                      "disagreements": []})
    for key, h in human.items():
        j = judged.get(key)
        if not j:
            continue
        fam = h["family"]
        d = per_family[fam]
        d["n"] += 1
        d["judge"].append(j["judge_letter"])
        d["human"].append(h["human_letter"])
        if j["judge_letter"] == h["human_letter"]:
            d["agree"] += 1
        else:
            d["disagreements"].append({
                "id": h["id"], "condition": h["condition"], "path": j["path"],
                "judge": j["judge_letter"], "judge_meaning": j.get("judge_meaning"),
                "human": h["human_letter"],
                "human_meaning": j["judge_option_map"].get(h["human_letter"]),
                "note": h.get("note", ""),
            })

    # GOAL.md asks for judge verdicts *and* direct-inspection verdicts in the same
    # per-family results file, so fold the inspection labels back in.
    for fam in os.listdir(IMG):
        p = os.path.join(IMG, fam, "results.jsonl")
        if not os.path.isfile(p):
            continue
        rows = [json.loads(l) for l in open(p) if l.strip()]
        changed = False
        for r in rows:
            h = human.get((r["id"], r["condition"]))
            if h:
                r["human_letter"] = h["human_letter"]
                r["human_meaning"] = r["judge_option_map"].get(h["human_letter"])
                r["human_note"] = h.get("note", "")
                r["judge_agrees"] = h["human_letter"] == r["judge_letter"]
                changed = True
        if changed:
            with open(p, "w") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    out = {}
    for fam, d in sorted(per_family.items()):
        out[fam] = {
            "n_inspected": d["n"],
            "raw_agreement": round(d["agree"] / d["n"], 4) if d["n"] else None,
            "cohens_kappa": (round(kappa(d["judge"], d["human"]), 4)
                             if kappa(d["judge"], d["human"]) is not None else None),
            "judge_label_counts": dict(Counter(d["judge"])),
            "human_label_counts": dict(Counter(d["human"])),
            "disagreements": d["disagreements"],
        }
    with open(os.path.join(ROOT, "pilot", "judge_agreement.json"), "w") as f:
        json.dump(out, f, indent=2)

    for fam, d in out.items():
        print(f"{fam:<16} n={d['n_inspected']:<4} agreement={d['raw_agreement']}  "
              f"kappa={d['cohens_kappa']}  ({len(d['disagreements'])} disagreements)")


if __name__ == "__main__":
    main()
