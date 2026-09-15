"""Merge the space-builder prompts (S) with the hand-written plain-mention
controls (P), locate the target span in both, and pick a control word.

Reads   data/generation/prompts.jsonl            (S: item_id, family, prompt, target)
        research/overreal/mechanism/prompts/plain_<family>.jsonl
            fields: item_id, plain, optional target_text / target_text_s /
            target_text_p (span text overrides; "|" separates several spans),
            optional exclude (reason string)
Writes  research/overreal/mechanism/pairs.jsonl
            item_id, family, target, s_text, s_spans, p_text, p_spans,
            c_word, c_spans_s, c_spans_p, exclude
        where spans are [[start, end], ...] character offsets.

Control word: a content word shared by S and P (case-insensitive), at least
four letters, not a stopword or cue word, not overlapping the target span,
and nearest to the target span in S. The "other-word probe" reads its span.

Usage: python build_pairs.py
"""
import json
import os
import re
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(ROOT, "..", "..", ".."))
S_PATH = os.path.join(REPO, "data", "generation", "prompts.jsonl")
P_DIR = os.path.join(ROOT, "prompts")
OUT = os.path.join(ROOT, "pairs.jsonl")

FAMILIES = ["cancellation", "attribution", "figurative", "perspectival"]

STOP = set("""
a an the and or but of to in on at by for with from into onto over under
above below behind beside between beyond near next off out up down through
across along around about after before during until while when where which
who whom whose what that this these those there here it its is are was were
be been being have has had do does did not no nor never none nothing any
some all every each both either neither one two three more most much many
few little less least very just only also too even still yet again once
than then so as if though although because since whether while his her
their our your my mine yours hers theirs ours him them us we you they she he
i me we am can could may might must shall should will would ought need
generate image photo photographed picture strictly completely entirely
""".split())

CUES = set("""
no without never not nothing none like resembles resembling resemble
reminiscent recalls recall evokes evoke believes believe believing thinks
think thinking remembers remember remembering imagines imagine imagining
dreams dream dreaming expects expect expecting wondering wonders wishing
wishes wish convinced envisions visualizing pretending hidden hides hiding
behind inside closed shut locked opaque covered face-down upside underneath
beneath back front camera sight printed prints shows showing shown displayed
displays displaying engraved contains containing reads says single anywhere
scene visible open opened closed drawn toward facing side cover surface
""".split())

WORD = re.compile(r"[A-Za-z][A-Za-z'\-]+")


def norm_target(t):
    return t.strip().strip('"').strip("'").strip().rstrip(".").strip()


def find_spans(text, span_text):
    spans = []
    for piece in span_text.split("|"):
        piece = piece.strip()
        m = re.search(re.escape(piece), text, flags=re.IGNORECASE)
        if not m:
            return None
        spans.append([m.start(), m.end()])
    return spans


def overlaps(a, b, spans):
    return any(a < e and b > s for s, e in spans)


def control_word(s_text, p_text, s_spans):
    p_words = {m.group(0).lower() for m in WORD.finditer(p_text)}
    best = None
    for m in WORD.finditer(s_text):
        w = m.group(0)
        wl = w.lower().strip("'-")
        if len(wl) < 4 or wl in STOP or wl in CUES or wl not in p_words:
            continue
        if overlaps(m.start(), m.end(), s_spans):
            continue
        dist = min(abs(m.start() - e) if m.start() >= e else abs(s - m.end())
                   for s, e in s_spans)
        if best is None or dist < best[0]:
            best = (dist, w, [m.start(), m.end()])
    if best is None:
        return None, None, None
    _, w, s_span = best
    pm = re.search(r"\b" + re.escape(w) + r"\b", p_text, flags=re.IGNORECASE)
    return w, [s_span], [[pm.start(), pm.end()]]


def main():
    s_rows = {}
    for line in open(S_PATH):
        if line.strip():
            r = json.loads(line)
            s_rows[r["item_id"]] = r

    p_rows = {}
    for fam in FAMILIES:
        for line in open(os.path.join(P_DIR, f"plain_{fam}.jsonl")):
            if line.strip():
                r = json.loads(line)
                p_rows[r["item_id"]] = r

    missing = [k for k in s_rows if k not in p_rows]
    extra = [k for k in p_rows if k not in s_rows]
    if missing or extra:
        raise SystemExit(f"missing P for {missing}\nP without S: {extra}")

    out = []
    problems = []
    for item_id, s in s_rows.items():
        p = p_rows[item_id]
        base = norm_target(s["target"])
        s_span_text = p.get("target_text_s") or p.get("target_text") or base
        p_span_text = p.get("target_text_p") or p.get("target_text") or base
        row = {
            "item_id": item_id,
            "family": s["family"],
            "target": s["target"],
            "s_text": s["prompt"],
            "p_text": p["plain"],
            "exclude": p.get("exclude"),
            "s_spans": None, "p_spans": None,
            "c_word": None, "c_spans_s": None, "c_spans_p": None,
        }
        if row["exclude"]:
            out.append(row)
            continue
        row["s_spans"] = find_spans(s["prompt"], s_span_text)
        row["p_spans"] = find_spans(p["plain"], p_span_text)
        if row["s_spans"] is None:
            problems.append((item_id, "S", s_span_text, s["prompt"]))
        if row["p_spans"] is None:
            problems.append((item_id, "P", p_span_text, p["plain"]))
        if row["s_spans"] is not None:
            row["c_word"], row["c_spans_s"], row["c_spans_p"] = control_word(
                s["prompt"], p["plain"], row["s_spans"])
        out.append(row)

    if problems:
        for item_id, side, span_text, text in problems:
            print(f"SPAN NOT FOUND {item_id} [{side}] {span_text!r} in: {text}")
        raise SystemExit(f"{len(problems)} span problems")

    with open(OUT, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_fam = defaultdict(list)
    for r in out:
        by_fam[r["family"]].append(r)
    print(f"{len(out)} items -> {OUT}")
    print(f"{'family':<14}{'n':>5}{'excl':>6}{'S words':>9}{'P words':>9}"
          f"{'|diff|>4':>10}{'no ctrl':>9}")
    for fam in FAMILIES:
        rows = [r for r in by_fam[fam] if not r["exclude"]]
        ex = sum(1 for r in by_fam[fam] if r["exclude"])
        sw = [len(r["s_text"].split()) for r in rows]
        pw = [len(r["p_text"].split()) for r in rows]
        big = sum(1 for a, b in zip(sw, pw) if abs(a - b) > 4)
        noc = sum(1 for r in rows if r["c_word"] is None)
        print(f"{fam:<14}{len(rows):>5}{ex:>6}{sum(sw)/len(sw):>9.1f}"
              f"{sum(pw)/len(pw):>9.1f}{big:>10}{noc:>9}")


if __name__ == "__main__":
    main()
