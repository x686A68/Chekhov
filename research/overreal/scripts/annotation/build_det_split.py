"""Build the OverReal-Det split from overreal_v1/metadata.jsonl.

One rule for both annotation waves (the marathon review was blind, so both
waves are independent double annotations): take the intersection of the two
label sets.
  non-empty, not {other}  -> gold      (the intersection is the gold label)
  empty                   -> ambiguous (both label sets released, no gold)
  {other}                 -> excluded  (both judged the image unusable)

Eligibility: the image has two annotations and its item is in OverReal-Gen
(included). Cancellation rows imported from the nb2 run carry a single
annotation source, so they join the gold set by construction but are excluded
from the agreement statistics.

Writes data/overreal_v1/det_split.jsonl: {image_id, split, gold_labels,
labels_1, labels_2, wave} and prints per-family statistics.
"""
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
META = ROOT / "data" / "overreal_v1" / "metadata.jsonl"
OUT = ROOT / "data" / "overreal_v1" / "det_split.jsonl"


def main():
    rows, stats = [], collections.defaultdict(collections.Counter)
    kappa_pairs = collections.defaultdict(list)
    for line in open(META, encoding="utf-8"):
        r = json.loads(line)
        if not r.get("annotated") or r.get("included") is False:
            continue
        l1, l2 = set(r["label_1"]), set(r["label_2"])
        if not l1 or not l2:
            continue
        wave = "marathon" if r.get("source_folder") is not None else "blind"
        self_pair = wave == "marathon" and r["family"] == "cancellation"
        inter = l1 & l2
        if inter == {"other"}:
            split, gold = "excluded", []
        elif inter:
            split, gold = "gold", sorted(inter - {"other"}) or sorted(inter)
        else:
            split, gold = "ambiguous", []
        rows.append({"image_id": r["image_id"], "split": split,
                     "gold_labels": gold, "labels_1": sorted(l1),
                     "labels_2": sorted(l2), "wave": wave})
        stats[r["family"]][split] += 1
        if not self_pair:
            kappa_pairs[r["family"]].append((frozenset(l1), frozenset(l2)))

    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def kappa(pairs):
        n = len(pairs)
        po = sum(bool(a & b) for a, b in pairs) / n
        dist = collections.Counter(s for ab in pairs for s in ab)
        t = sum(dist.values())
        pe = sum(c1 * c2 * bool(s1 & s2) for s1, c1 in dist.items()
                 for s2, c2 in dist.items()) / t ** 2
        return po, (po - pe) / (1 - pe)

    total = collections.Counter()
    all_pairs = []
    for fam in ["cancellation", "attribution", "figurative", "perspectival"]:
        s = stats[fam]
        po, k = kappa(kappa_pairs[fam])
        print(f"{fam:14s} gold={s['gold']:5d} amb={s['ambiguous']:4d} "
              f"excl={s['excluded']:3d} | agree(po)={po:.3f} kappa={k:.3f} "
              f"(n={len(kappa_pairs[fam])})")
        total.update(s)
        all_pairs += kappa_pairs[fam]
    po, k = kappa(all_pairs)
    print(f"{'TOTAL':14s} gold={total['gold']:5d} amb={total['ambiguous']:4d} "
          f"excl={total['excluded']:3d} | agree(po)={po:.3f} kappa={k:.3f} "
          f"(n={len(all_pairs)})")


if __name__ == "__main__":
    main()
