"""Phase 0 — score the text pilot.

Two scores per generation:

  realized_surface      the entity string occurs at all (GOAL.md's specified scoring:
                        word-form-normalized string match)
  realized_affirmative  the entity is realized *as present in the scene*, i.e. at least
                        one occurrence is not inside the scope of the family's marking
                        device (negation, attribution, simile, occlusion, word-mention)

Both are reported. The surface score is what GOAL.md asks for, but on its own it is
confounded for exactly the families the construct is about: "the lobby contained no
elephant" is a surface hit and a correct suppression. The affirmative score is a
regex-scope heuristic, so every family's disagreement count is reported too and the
disagreeing generations are dumped for manual inspection.

Rule 3 is applied here: an (entity, scenario) pair whose entity appears spontaneously
in the A condition is unusable. Summaries are written both unfiltered and filtered.
"""
import json
import os
import re
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "pilot", "text", "raw")
OUT = os.path.join(ROOT, "pilot", "text")

PLURALS = {"elephant": "elephants", "snake": "snakes", "tiger": "tigers", "wolf": "wolves"}

# Content keys for family 4b: the message counts as realized if its content is reported,
# not only if it is quoted verbatim.
MSG_KEYS = {
    "I'm sorry": [r"\bsorry\b", r"\bapolog"],
    "the meeting is cancelled": [r"\bcancell?ed\b", r"\bcancellation\b", r"\bmeeting is off\b"],
    "flight 402 at nine": [r"\b402\b", r"\bflight\b.{0,20}\bnine\b"],
}

WINDOW = 60  # chars of left context searched for a marking cue
RWINDOW = 45  # chars of right context — "the elephant moved behind it" marks after the fact

NEG_CUES = r"(?:\bno\b|\bnot\b|\bnone\b|\bnever\b|\bwithout\b|\babsence\b|\babsent\b|\bdevoid\b|\bfree (?:of|from)\b|\bempty of\b|\black(?:s|ing)?\b|\bisn't\b|\baren't\b|\bwasn't\b|\bweren't\b|\bnothing\b|\bnor\b|\bunlike\b|\buntouched by\b|\bno sign of\b|\bbereft of\b|\bunmarked by\b|\bundisturbed by\b)"
ATTR_CUES = r"(?:\bbelie(?:f|v)|\bthought\b|\bthink|\bimagin|\bmistook\b|\bmistak|\bconvinc|\bseem|\bappear|\bas if\b|\bas though\b|\bsupposed?\b|\bclaim|\binsist|\bin (?:her|his|their) mind\b|\bswore\b|\bsure\b|\bcertain\b|\bwhat she took\b|\bwhat he took\b|\bperceiv|\bfanc(?:y|ied)\b|\bpresence of what|\bcould ?n[o']t tell\b|\bwonder|\bwhether\b|\bor an?\b|\bmight\b|\bsuspect)"
# The quality-noun list covers "the ferocity of a wolf" — a quality taken from the
# entity, not the entity itself. It deliberately excludes "presence of", which in the
# P condition marks a real referent.
FIG_CUES = (r"(?:\blike\b|\bas [a-z]+ as\b|\bresembl|\breminiscent\b|\bmetaphor|\bsimile\b"
            r"|\bcompar|\bas if\b|\bmight (?:as well )?have been\b|\bsomething of\b|\brival"
            r"|\bevok|\bimpression\b|\bsuggest"
            r"|\b(?:grace|ferocity|hunger|strength|weight|bulk|size|air|manner|quality|sense"
            r"|shadow|memory|patience|instinct|stealth|silence|bearing|gait)\s+of\b)")
OCC_CUES = r"(?:\bhidden\b|\bhides?\b|\bhiding\b|\bbehind\b|\bconceal|\bobscur|\bout of sight\b|\bunseen\b|\bnot visible\b|\binvisible\b|\bcannot see\b|\bcan't see\b|\bcould not see\b|\bcouldn't see\b|\bbeyond\b|\bblocked\b|\bblocks\b|\bscreen(?:ed|s)? (?:off|from)\b)"
# deliberately excludes container nouns (crate, sign): they co-occur with the target in
# the P condition too, where the target really is realized as a referent
MENTION_CUES = r"(?:\bword\b|\bwords\b|\bstencill?ed\b|\bstencil\b|\bletter(?:s|ing)?\b|\blabell?ed\b|\blabel\b|\bprinted\b|\bpainted\b|\bspell|\bstamp|\bmarking(?:s)?\b|\breads?\b|\bwritten\b|\bblock capitals\b|\btext\b|\bsignage\b)"

FAMILY_CUES = {
    "1_existence": NEG_CUES,
    "2_attribution": f"(?:{ATTR_CUES}|{NEG_CUES})",
    "3_figurative": f"(?:{FIG_CUES}|{NEG_CUES})",
    "4a_occlusion": f"(?:{OCC_CUES}|{NEG_CUES})",
    "4b_legibility": f"(?:{OCC_CUES}|{NEG_CUES}|\\bcannot (?:read|make out)\\b|\\bcan't (?:read|make out)\\b|\\bunable to\\b|\\billegible\\b|\\bunreadable\\b|\\bfaces? away\\b|\\bturned away\\b|\\bangled away\\b)",
    "5_use_mention": f"(?:{MENTION_CUES}|{NEG_CUES})",
    "6_relevance": NEG_CUES,
}


def mentions(text, row):
    """Yield (start, end) spans of the target in the generation."""
    if row["family"] == "4b_legibility":
        pats = MSG_KEYS[row["entity"]] + [re.escape(row["entity"])]
    else:
        e = row["entity"]
        pats = [rf"\b{e}s?\b", rf"\b{PLURALS.get(e, e)}\b"]
    spans = []
    for p in pats:
        for m in re.finditer(p, text, flags=re.I):
            spans.append((m.start(), m.end()))
    return sorted(set(spans))


def score(row):
    text = row["output"]
    spans = mentions(text, row)
    surface = bool(spans)
    cue = FAMILY_CUES[row["family"]]
    affirmative = False
    marked_spans = 0
    for s, e in spans:
        left = text[max(0, s - WINDOW):s]
        right = text[e:e + RWINDOW]
        # a mention set in the target's own uppercase form is a word-mention (family 5)
        as_word = row["family"] == "5_use_mention" and text[s:e].isupper()
        quoted = text[max(0, s - 2):s].strip().endswith(('"', "'", "“"))
        # family 3 only: "wolf-like", "a wolf's hunger" are modifier uses, not referents.
        # The same test would misfire on family 4a, where "the wolf's head peeking over
        # the wall" is a genuine over-realization.
        morph = (row["family"] == "3_figurative"
                 and (right[:2] in ("'s", "’s") or right[:1] == "-" or text[max(0, s - 1):s] == "-"))
        if (re.search(cue, left, flags=re.I) or re.search(cue, right, flags=re.I)
                or as_word or quoted or morph):
            marked_spans += 1
        else:
            affirmative = True
    return surface, affirmative, len(spans), marked_spans


def main():
    rows = []
    for fn in sorted(os.listdir(RAW)):
        if fn.endswith(".jsonl"):
            with open(os.path.join(RAW, fn)) as f:
                rows += [json.loads(l) for l in f if l.strip()]
    if not rows:
        print("no raw generations yet")
        return

    for r in rows:
        s, a, n, m = score(r)
        r["realized_surface"] = s
        r["realized_affirmative"] = a
        r["n_mentions"] = n
        r["n_marked_mentions"] = m
        r["realized"] = a  # headline flag = affirmative realization

    # --- rule 3: drop (entity, scenario) pairs that realize spontaneously under A ------
    a_hits = defaultdict(list)
    for r in rows:
        if r["condition"] == "A" and r["realized_affirmative"]:
            a_hits[(r["family"], r["id"])].append(r["model"])
    dropped = {f"{k[0]}/{k[1]}": v for k, v in a_hits.items()}
    with open(os.path.join(OUT, "dropped_items.json"), "w") as f:
        json.dump({"reason": "entity realized spontaneously under condition A (GOAL.md rule 3)",
                   "dropped": dropped, "n_dropped": len(dropped)}, f, indent=2)

    by_family = defaultdict(list)
    for r in rows:
        by_family[r["family"]].append(r)

    disagree = []
    overview = {}
    for fam, frows in sorted(by_family.items()):
        d = os.path.join(OUT, fam)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "results.jsonl"), "w") as f:
            for r in sorted(frows, key=lambda x: (x["model"], x["id"], x["condition"])):
                f.write(json.dumps({k: r[k] for k in (
                    "model", "id", "family", "entity", "scenario_id", "scenario", "device",
                    "condition", "prompt", "output", "realized", "realized_surface",
                    "realized_affirmative", "n_mentions", "n_marked_mentions")},
                    ensure_ascii=False) + "\n")
                if r["realized_surface"] != r["realized_affirmative"]:
                    disagree.append(r)

        summary = {"family": fam, "n_items": len({r["id"] for r in frows}), "by_model": {},
                   "by_model_entity": {}}
        for filt in (False, True):
            key = "filtered" if filt else "unfiltered"
            for model in sorted({r["model"] for r in frows}):
                sel = [r for r in frows if r["model"] == model
                       and not (filt and (fam, r["id"]) in a_hits)]
                summary["by_model"].setdefault(model, {})[key] = cell_stats(sel)
                for ent in sorted({r["entity"] for r in sel}):
                    esel = [r for r in sel if r["entity"] == ent]
                    summary["by_model_entity"].setdefault(model, {}).setdefault(ent, {})[key] = cell_stats(esel)
        with open(os.path.join(d, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        overview[fam] = summary["by_model"]

    with open(os.path.join(OUT, "overview.json"), "w") as f:
        json.dump(overview, f, indent=2)
    with open(os.path.join(OUT, "surface_vs_affirmative_disagreements.jsonl"), "w") as f:
        for r in disagree:
            f.write(json.dumps({k: r[k] for k in ("model", "id", "family", "entity",
                                                  "condition", "output", "n_mentions",
                                                  "n_marked_mentions")}, ensure_ascii=False) + "\n")

    print(f"scored {len(rows)} generations across {len(by_family)} family cells")
    print(f"rule-3 dropped items: {len(dropped)}")
    print(f"surface/affirmative disagreements: {len(disagree)}")
    print()
    hdr = f"{'family':<16}{'model':<14}" + "".join(f"{c:>9}" for c in ("S", "P", "A", "delta", "S_surf", "P_surf"))
    print(hdr)
    for fam, per_model in overview.items():
        for model, d in per_model.items():
            u = d["filtered"]
            print(f"{fam:<16}{model:<14}"
                  f"{u['S']:>9.2f}{u['P']:>9.2f}{u['A']:>9.2f}{u['delta']:>9.2f}"
                  f"{u['S_surface']:>9.2f}{u['P_surface']:>9.2f}")


def cell_stats(sel):
    def rate(cond, field="realized_affirmative"):
        s = [r for r in sel if r["condition"] == cond]
        return (sum(r[field] for r in s) / len(s)) if s else float("nan")
    out = {c: round(rate(c), 4) for c in "SPA"}
    out.update({f"{c}_surface": round(rate(c, "realized_surface"), 4) for c in "SPA"})
    out["delta"] = round(out["P"] - out["S"], 4)
    out["delta_surface"] = round(out["P_surface"] - out["S_surface"], 4)
    out["n_per_cell"] = len([r for r in sel if r["condition"] == "S"])
    return out


if __name__ == "__main__":
    main()
