"""Family 1 (existence-canceling) — build S-condition candidates from NegBench pairs.

Design (mirrored in dataset/family1/DESIGN.md, which is the paper-facing writeup):

We inherit NegBench's (scene, absent-object) PAIRS, not its sentences. Each row of
NegBench's COCO_val_retrieval.csv carries, for one COCO val2017 image:

    positive_objects  COCO-80 objects annotated as present in the image
    negative_objects  objects proposed as related-but-absent, then verified absent
                      by NegBench's detector filter
    captions          the five original human-written COCO captions

The five COCO captions are already in text-to-image prompt register; NegBench's own
negated captions are in caption/retrieval register and are discarded. An item is

    prompt = clean(caption) + D(target)      target = one absent object

where D is drawn from a small bank of trailing negation phrases built on the family's
cue words (*no*, *without*). The target string appears in the prompt verbatim, so the
presence judge and the prompt can never disagree about the surface form.

Two plausibility bins, 50/50:
  plausible    target from negative_objects — semantically related to the scene and
               detector-verified absent (NegBench's two-step validation, inherited)
  implausible  target re-paired by us: a COCO-80 object with ZERO co-occurrence with
               every positive object of that image across all 5,000 val images, and a
               global frequency >= MIN_GLOBAL so the zero is informative

Filters: caption length in words, no meta-language (photo/image/camera...), no
pre-existing negation, target not mentioned in the caption (word-boundary, plural
tolerated, compound classes masked first: "hot dog" does not count as "dog").
Quotas: one item per image; per-target cap within each bin. Deterministic seed.

Output: dataset/family1/f1_S_candidates_v1.jsonl  (+ .stats.json)
The scenario (cleaned caption) is retained per item so P = Sc + "with a(n) E" and
A = Sc can be derived mechanically later; only S is built here.
"""
import ast
import csv
import json
import os
import random
import re
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
SRC = os.path.join(BASE, "dataset", "family1", "source", "COCO_val_retrieval.csv")
OUT_DIR = os.path.join(BASE, "dataset", "family1")
OUT = os.path.join(OUT_DIR, "f1_S_candidates_v1.jsonl")
STATS = os.path.join(OUT_DIR, "f1_S_candidates_v1.stats.json")

SEED = 20260814
N_TOTAL = 500          # candidates; human pass trims to the final set
N_PER_BIN = N_TOTAL // 2
CAP_PER_TARGET = 5     # per (target, bin); raised stepwise if a bin cannot fill
MIN_WORDS, MAX_WORDS = 6, 18
MIN_GLOBAL = 25        # a zero co-occurrence only counts if the class itself is common

COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]
# excluded as targets: number-defective ("no person" wants "people", scissors/skis are
# plural-only) — the negation phrase and the stored target would diverge
TARGET_EXCLUDE = {"person", "scissors", "skis"}
TARGET_POOL = [c for c in COCO80 if c not in TARGET_EXCLUDE]

# trailing devices on the family's cue words; every template keeps the target verbatim
TEMPLATES = {
    "T1": ("no", "{cap}, with no {e} in sight"),
    "T2": ("without", "{cap}, without a single {e}"),
    "T3": ("no", "{cap}, no {e} anywhere in the scene"),
    "T4": ("no", "{cap}, and not a single {e}"),
    "T5": ("without", "{cap}, without any {e}"),
}

META_WORDS = re.compile(
    r"\b(image|picture|photo|photograph|photographed|photographer|camera|"
    r"close[- ]?up|black and white|closeup)\b", re.I)
NEGATION = re.compile(r"\b(no|not|without|never|none|nothing|empty|missing)\b|n't", re.I)
STARTS_BAD = re.compile(r"^\s*(there (is|are)|this is|it is|an? (image|picture|photo))\b", re.I)


def clean_caption(c):
    """Return cleaned caption or None if rejected."""
    c = re.sub(r"\s+", " ", c).strip()
    if not c or STARTS_BAD.search(c) or META_WORDS.search(c) or NEGATION.search(c):
        return None
    n = len(c.split())
    if not (MIN_WORDS <= n <= MAX_WORDS):
        return None
    if not re.match(r"^[A-Za-z]", c):
        return None
    c = c.rstrip(" .")
    if re.search(r"[.!?;:]$", c):  # multi-sentence leftovers
        return None
    return c[0].lower() + c[1:]


COMPOUNDS = [c for c in COCO80 if " " in c]


def mentions(caption, target):
    """Word-boundary mention check, plural-tolerant, compounds masked first."""
    text = caption.lower()
    for comp in COMPOUNDS:
        if comp != target and target in comp:
            text = text.replace(comp, " ")
    return re.search(rf"\b{re.escape(target)}(s|es)?\b", text) is not None


def load_rows():
    rows = []
    with open(SRC, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(dict(
                image_id=int(r["image_id"]),
                positives=[p.lower() for p in ast.literal_eval(r["positive_objects"])],
                negatives=[n.lower() for n in ast.literal_eval(r["negative_objects"])],
                captions=ast.literal_eval(r["captions"]),
            ))
    return rows


def build():
    rng = random.Random(SEED)
    rows = load_rows()

    # sanity: NegBench negatives should live inside COCO-80
    stray = {n for r in rows for n in r["negatives"]} - set(COCO80)
    if stray:
        print(f"note: {len(stray)} non-COCO80 negatives ignored: {sorted(stray)[:10]}")

    # co-occurrence of positives across all images, and global frequency
    freq = Counter()
    cooc = defaultdict(Counter)
    for r in rows:
        ps = sorted(set(r["positives"]))
        freq.update(ps)
        for i, a in enumerate(ps):
            for b in ps[i + 1:]:
                cooc[a][b] += 1
                cooc[b][a] += 1

    order = list(range(len(rows)))
    rng.shuffle(order)
    used_images = set()
    items = []
    tmpl_ids = list(TEMPLATES)

    def viable_captions(r):
        out = []
        for ci, c in enumerate(r["captions"]):
            cc = clean_caption(c)
            if cc:
                out.append((ci, cc))
        return out

    def emit(r, ci, cap, target, bin_name):
        tid = rng.choice(tmpl_ids)
        cue, tpl = TEMPLATES[tid]
        items.append(dict(
            id=f"f1_nb_{len(items):03d}",
            family="1_existence",
            prompt=tpl.format(cap=cap, e=target),
            target=target,
            cue=cue,
            template_id=tid,
            plausibility=bin_name,
            scenario=cap,
            negbench_image_id=r["image_id"],
            caption_index=ci,
            source="negbench:COCO_val_retrieval.csv",
        ))
        used_images.add(r["image_id"])

    def fill(bin_name, candidates_for):
        count, cap_per_target = 0, CAP_PER_TARGET
        target_use = Counter()
        while count < N_PER_BIN and cap_per_target <= 12:
            for idx in order:
                if count >= N_PER_BIN:
                    break
                r = rows[idx]
                if r["image_id"] in used_images:
                    continue
                caps = viable_captions(r)
                if not caps:
                    continue
                ci, cap = rng.choice(caps)
                cands = [e for e in candidates_for(r)
                         if target_use[e] < cap_per_target and not mentions(cap, e)]
                if not cands:
                    continue
                # least-used target first keeps the pool diverse
                e = min(cands, key=lambda x: (target_use[x], rng.random()))
                emit(r, ci, cap, e, bin_name)
                target_use[e] += 1
                count += 1
            cap_per_target += 1  # relax only if a full sweep could not fill the bin
        return count

    def plausible_cands(r):
        return [e for e in r["negatives"] if e in TARGET_POOL]

    def implausible_cands(r):
        if len(r["positives"]) < 2:
            return []
        ps = set(r["positives"])
        return [e for e in TARGET_POOL
                if e not in ps and freq[e] >= MIN_GLOBAL
                and all(cooc[e][p] == 0 for p in ps)]

    n_pl = fill("plausible", plausible_cands)
    n_im = fill("implausible", implausible_cands)

    rng.shuffle(items)
    for i, it in enumerate(items):
        it["id"] = f"f1_nb_{i:03d}"

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")

    stats = dict(
        seed=SEED, n_items=len(items), plausible=n_pl, implausible=n_im,
        templates=dict(Counter(i["template_id"] for i in items)),
        cues=dict(Counter(i["cue"] for i in items)),
        distinct_targets=len({i["target"] for i in items}),
        top_targets=Counter(i["target"] for i in items).most_common(10),
        mean_prompt_words=round(sum(len(i["prompt"].split()) for i in items) / len(items), 1),
        distinct_images=len({i["negbench_image_id"] for i in items}),
    )
    with open(STATS, "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    build()
