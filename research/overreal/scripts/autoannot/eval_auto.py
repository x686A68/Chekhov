"""Score an automatic-annotation run against the human gold.

  python eval_auto.py data/overreal_v1/auto/v1__qwen25-vl-7b-instruct.jsonl --split dev

Reports, per family and per (family, generator): n, accuracy, Cohen's kappa
on the four labels, kappa on the coarse split (over-realized {D,S} vs correct
{I,W}), and presence accuracy (Q1 against gold-in-{D,S,I}). Also prints the
confusion matrix per family (rows gold, columns auto). A gold image whose
intersection carries two labels counts the auto label as correct if it is in
the set; otherwise the first gold label stands.

--amb: on the ambiguous split, count which annotator's label set the auto
label falls in.
"""
import argparse
import collections
import json
import sys

from common import FAMILIES, LABELS, derive_label, load_meta, load_split

COARSE = {"disruptive": "over", "silent": "over", "integrated": "ok", "withheld": "ok",
          "other": "other"}


def kappa(pairs):
    n = len(pairs)
    if n == 0:
        return float("nan")
    po = sum(a == b for a, b in pairs) / n
    ca = collections.Counter(a for a, _ in pairs)
    cb = collections.Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def gold_of(r, auto):
    g = r["gold_labels"]
    return auto if auto in g else g[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--amb", action="store_true")
    ap.add_argument("--by-generator", action="store_true")
    ap.add_argument("--show-errors", type=int, default=0)
    ap.add_argument("--family", default="")
    args = ap.parse_args()

    auto = {}
    for line in open(args.run, encoding="utf-8"):
        r = json.loads(line)
        r["label"] = derive_label(r["answers"], r["questions"])
        auto[r["image_id"]] = r
    meta = load_meta()
    split = [r for r in load_split() if r["split"] == args.split]
    rows = [r for r in split if r["image_id"] in auto]
    if args.family:
        rows = [r for r in rows if r["family"] == args.family]
    print(f"{len(rows)} of {len(split)} {args.split} images annotated in {args.run}")

    if args.amb:
        c = collections.Counter()
        for r in rows:
            a = auto[r["image_id"]]["label"]
            in1, in2 = a in r["labels_1"], a in r["labels_2"]
            c[(r["family"], "ann1" if in1 and not in2 else "ann2" if in2 and not in1
               else "both" if in1 else "neither" if a else "unparsed")] += 1
        for fam in FAMILIES:
            print(f"{fam:14s} " + " ".join(f"{k}={c[(fam,k)]:3d}" for k in
                                          ["ann1", "ann2", "both", "neither", "unparsed"]))
        return

    def report(name, rs):
        pairs, coarse, pres, unparsed = [], [], [], 0
        for r in rs:
            a = auto[r["image_id"]]["label"]
            if a is None:
                unparsed += 1
                continue
            g = gold_of(r, a)
            pairs.append((g, a))
            coarse.append((COARSE[g], COARSE[a]))
            q1 = auto[r["image_id"]]["answers"].get("Q1")
            if q1 is not None:
                pres.append((g != "withheld", q1))
        n = len(pairs)
        acc = sum(a == b for a, b in pairs) / n if n else float("nan")
        pacc = sum(a == b for a, b in pres) / len(pres) if pres else float("nan")
        print(f"{name:40s} n={n:4d} acc={acc:.3f} k4={kappa(pairs):.3f} "
              f"k2={kappa(coarse):.3f} presence={pacc:.3f} unparsed={unparsed}")
        return pairs

    allpairs = []
    for fam in FAMILIES:
        rs = [r for r in rows if r["family"] == fam]
        if not rs:
            continue
        pairs = report(fam, rs)
        allpairs += pairs
        if args.by_generator:
            gens = sorted({(r["generator"], r["prompt_cond"]) for r in rs}, key=str)
            for g in gens:
                sub = [r for r in rs if (r["generator"], r["prompt_cond"]) == g]
                report(f"  {fam}/{g[0]}/{g[1]}", sub)
        cm = collections.Counter(pairs)
        cols = LABELS + (["other"] if any(a == "other" for _, a in pairs) else [])
        print("   gold\\auto " + " ".join(f"{l[:4]:>5s}" for l in cols))
        for g in LABELS:
            print(f"   {g[:10]:10s} " + " ".join(f"{cm[(g,a)]:5d}" for a in cols))
    print(f"{'ALL':40s} n={len(allpairs):4d} acc={sum(a==b for a,b in allpairs)/max(1,len(allpairs)):.3f} "
          f"k4={kappa(allpairs):.3f} k2={kappa([(COARSE[a],COARSE[b]) for a,b in allpairs]):.3f}")

    if args.show_errors:
        shown = 0
        for r in rows:
            a = auto[r["image_id"]]
            if a["label"] and a["label"] not in r["gold_labels"]:
                m = meta[r["image_id"]]
                print(f"\n[{r['image_id']}] gold={r['gold_labels']} auto={a['label']} "
                      f"answers={a['answers']}\n  target={m['target']!r}\n  prompt={m['prompt'][:160]}")
                shown += 1
                if shown >= args.show_errors:
                    break


if __name__ == "__main__":
    main()
