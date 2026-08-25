"""Add the Family 1 (existence-canceling) split to data/overreal_v1 as `cancellation`.

Samples 50 items from the Nano Banana 2 run, prioritising the images where the
generator failed (over-realized the target), and writes them in the same layout the
other three families use:

    images/cancellation/prompt_NNNN/image_000.jpg   one gemini image per item
    metadata.jsonl                                   50 appended rows, same 14 keys

Selection: all 22 items with at least one over-realization (the failing sample is the
one included; s0 if both samples failed), plus 28 correctly-suppressed items chosen so
the split comes out balanced 25/25 across the plausible / implausible bins.

Label mapping onto the shared outcome vocabulary (paper 2.2):
  realized=False -> withheld     target left out, picture coherent
  realized=True  -> disruptive   the target is present although the prompt cancelled
                                 its existence, so the picture contradicts the prompt

Family 1 admits no other failure label. `silent` requires the realization to leave the
picture consistent with what was asked; under an explicit "no X" device any X at all
puts the image in contradiction with its prompt, however incidental the object is.
That includes the two depiction-only cases (sheep on a TV screen, a traffic light in a
framed photograph): DECISIONS.md #43 fixes family 1's criterion as holding in any
style, so unlike the marking families 2 and 3 there is no depicted form under which
the target may legitimately appear. `integrated` cannot arise for the same reason.

Provenance differs from the other families and is recorded honestly: generator is
gemini-3.1-flash-image (not 2.5), annotation is one model rater at normal viewing
scale, and there is no second annotator, so `agreement` is null. Bin, confidence and
the annotator's evidence sentence go to cancellation_extra.jsonl rather than into
metadata.jsonl, which keeps its exact key set.
"""
import csv
import glob
import json
import os
import random
import shutil

ROOT = "/home/jiahao_huang/Chekhov"
DS = os.path.join(ROOT, "data", "overreal_v1")
ANN = os.path.join(ROOT, "research/overreal/dataset/family1/annotations_nb2.jsonl")
CHUNKS = os.path.join(ROOT, "research/overreal/dataset/family1/images/inspect_chunks")
IMG_DIR = os.path.join(DS, "images", "cancellation")
META = os.path.join(DS, "metadata.jsonl")
EXTRA = os.path.join(DS, "cancellation_extra.jsonl")
AUDIT = os.path.join(DS, "audit_log.csv")
SEED = 20260814
N_TOTAL, N_PER_BIN = 50, 25
GENERATOR = "gemini-3.1-flash-image"
ANNOTATOR = "claude-subagent"


def main():
    ann = [json.loads(l) for l in open(ANN)]
    paths = {}
    for f in sorted(glob.glob(os.path.join(CHUNKS, "chunk_*.jsonl"))):
        for l in open(f):
            r = json.loads(l)
            paths[r["key"]] = r["path"]

    by_item = {}
    for r in ann:
        by_item.setdefault(r["item_id"], []).append(r)
    for v in by_item.values():
        v.sort(key=lambda r: r["sample"])

    failed = {k: v for k, v in by_item.items() if any(r["realized"] for r in v)}
    clean = {k: v for k, v in by_item.items() if k not in failed}

    chosen = []           # (item_id, annotation row for the image to include)
    for k in sorted(failed):
        rows = failed[k]
        chosen.append((k, next(r for r in rows if r["realized"])))

    rng = random.Random(SEED)
    have = {"plausible": 0, "implausible": 0}
    for _, r in chosen:
        have[r["plausibility"]] += 1
    for b in ("plausible", "implausible"):
        need = N_PER_BIN - have[b]
        pool = sorted(k for k, v in clean.items() if v[0]["plausibility"] == b)
        if need > len(pool):
            raise SystemExit(f"{b}: need {need}, pool {len(pool)}")
        for k in rng.sample(pool, need):
            chosen.append((k, rng.choice(clean[k])))
    if len(chosen) != N_TOTAL:
        raise SystemExit(f"selected {len(chosen)}, expected {N_TOTAL}")

    chosen.sort(key=lambda t: t[0])

    if os.path.isdir(IMG_DIR):
        shutil.rmtree(IMG_DIR)
    os.makedirs(IMG_DIR)

    meta_rows, extra_rows = [], []
    for i, (item_id, r) in enumerate(chosen, start=1):
        pid = f"prompt_{i:04d}"
        src = paths[r["key"]]
        ext = os.path.splitext(src)[1]
        os.makedirs(os.path.join(IMG_DIR, pid))
        rel = f"images/cancellation/{pid}/image_000{ext}"
        shutil.copy2(src, os.path.join(DS, rel))
        label = "disruptive" if r["realized"] else "withheld"
        run = "nb2_full" if "nb2_full" in src else "nb2_pilot"
        meta_rows.append(dict(
            file_name=rel,
            image_id=f"cancellation/{pid}/image_000",
            family="cancellation",
            item_id=f"cancellation/{pid}",
            source_folder=run,
            annotator=ANNOTATOR,
            prompt=r["prompt"],
            target=r["target"],
            label_1=[label],
            label_2=[label],
            agreement=None,
            included=True,
            generator=GENERATOR,
            raw_path=os.path.relpath(src, ROOT),
        ))
        extra_rows.append(dict(
            image_id=f"cancellation/{pid}/image_000", source_key=r["key"],
            plausibility=r["plausibility"], sample=r["sample"],
            realized=r["realized"], confidence=r["confidence"],
            depiction_only=r["depiction_only"], evidence=r["evidence"],
            protocol="normal-scale single view, no magnification; single model rater",
        ))

    keys = list(json.loads(open(META).readline()).keys())
    assert list(meta_rows[0].keys()) == keys, (keys, list(meta_rows[0].keys()))

    with open(META, "a") as f:
        for r in meta_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(EXTRA, "w") as f:
        for r in extra_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(AUDIT, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family_added", "cancellation", "", "",
                    f"50 items sampled from the {GENERATOR} family-1 run; all 22 items "
                    f"with an over-realization included, 28 suppressions added to balance "
                    f"the plausible/implausible bins 25/25"])
        w.writerow(["label_mapping", "cancellation", "", "",
                    "binary presence annotation mapped to the shared vocabulary: "
                    "realized -> disruptive, suppressed -> withheld"])
        w.writerow(["label_judgment", "cancellation", "", "",
                    "every realization is disruptive: under an explicit 'no X' device the "
                    "presence of X contradicts the prompt itself, however incidental the "
                    "object is, so silent cannot arise in this family"])
        w.writerow(["label_judgment", "cancellation", "", "",
                    "the two depiction-only realizations (sheep on a TV screen, traffic "
                    "light in a framed photograph) are disruptive too: DECISIONS #43 fixes "
                    "family 1's criterion as style-independent, so there is no depicted "
                    "form under which the target may legitimately appear"])
        w.writerow(["single_annotator", "cancellation", "", "",
                    "one model rater, no second annotator, so agreement is null for all "
                    "50 rows; not comparable to the human-annotated families"])
        w.writerow(["generator_differs", "cancellation", "", "",
                    f"{GENERATOR}, whereas the other families use gemini-2.5-flash-image"])

    n_dis = sum(1 for r in meta_rows if r["label_2"] == ["disruptive"])
    print(f"wrote {len(meta_rows)} items -> {IMG_DIR}")
    print(f"  disruptive {n_dis}  withheld {len(meta_rows) - n_dis}")
    print(f"  plausible {sum(1 for r in extra_rows if r['plausibility'] == 'plausible')}"
          f"  implausible {sum(1 for r in extra_rows if r['plausibility'] == 'implausible')}")


if __name__ == "__main__":
    main()
