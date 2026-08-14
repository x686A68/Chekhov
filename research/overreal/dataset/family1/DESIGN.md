# Family 1 (existence-canceling) — S-condition construction from NegBench

Paper-facing writeup of how the Family 1 candidate pool was built. The build is fully
rule-based and deterministic: `scripts/build_family1_negbench.py`, seed 20260814.
Companion decision: DECISIONS.md #44.

## What is inherited, and what is not

Family 1 is the classic negation family; the paper includes it for completeness, not
novelty, and its materials are correspondingly inherited from the existing negation
benchmark **NegBench** (Alhamoud et al., CVPR 2025). What we inherit is the
**(scene, absent-object) pairs**, not the sentences:

- Source file: `source/COCO_val_retrieval.csv` from the NegBench release
  (sha256 `dfe92173...6de8d547`, 5,000 rows, one per COCO val2017 image). Each row
  carries `positive_objects` (COCO-80 objects annotated present), `negative_objects`
  (objects proposed by an LLM as *related but absent*, then verified absent by
  NegBench's detector filter), and the five original human-written COCO captions.
- NegBench's own negated captions are **discarded**: they are written in
  caption/retrieval register ("This image features A but not B"), which is the wrong
  register for a text-to-image prompt. The original COCO captions, by contrast, are
  already in prompt register and serve as the scene description.

The value of the inherited pairs is their two-step validation: the absent object is
semantically related to the scene (LLM proposal) and certifiably absent from it
(detector check). That is exactly the "plausible but absent" condition that makes an
existence-canceling item non-trivial.

## Item structure

Each item is `prompt` + `target`:

```
prompt = clean(caption) + D(target)
```

where `D` is one of five trailing negation phrases built on the family's cue words
(*no*, *without*):

| id | cue | template |
|----|-----|----------|
| T1 | no | ", with no {E} in sight" |
| T2 | without | ", without a single {E}" |
| T3 | no | ", no {E} anywhere in the scene" |
| T4 | no | ", and not a single {E}" |
| T5 | without | ", without any {E}" |

Only trailing attachment is used — mid-sentence embedding would require syntactic
surgery on found captions, with an error rate the 500-item scale does not justify. The
bank has five surface forms so that no single trailing phrase becomes a corpus-wide
stylistic artifact (which the detection half's mismatched-prompt control would
otherwise pick up). The target string appears in the prompt **verbatim** (singular
lemma), so the presence judge and the prompt can never disagree on surface form.

The cleaned caption is retained per item as `scenario`, so the paired conditions
derive mechanically when needed: `P = scenario + ", with a(n) {E}"`, `A = scenario`.
Only S is built here.

## Plausibility as a controlled factor

500 candidates, 250 per bin:

- **plausible** — target drawn from the row's `negative_objects`: related to the
  scene, absent from it (inherited validation, above).
- **implausible** — target re-paired by us: a COCO-80 class with **zero co-occurrence**
  with *every* positive object of that image, computed over all 5,000 val images, and
  global frequency >= 25 so that the zero is informative rather than an artifact of a
  rare class. This bin is NegBench's vocabulary but our pairing; NegBench does not
  provide incongruous negatives.

The two bins are analyzed separately: the base rate of coincidental realization (A
condition) is nonzero only in the plausible bin, and the Chekhov's-gun effect is
expected to differ across them.

## Filters and quotas

Captions: 6–18 words; no meta-language (image/picture/photo/camera/close-up...); no
pre-existing negation (no/not/without/never/none/nothing/empty/missing/n't); no
"there is/are" openers; single sentence; first letter lowercased; trailing period
stripped. Targets: COCO-80 minus number-defective classes (*person*, *scissors*,
*skis*, whose natural negation diverges from the stored lemma); word-boundary,
plural-tolerant mention check against the caption, with compound classes masked first
("hot dog" does not count as a mention of "dog"). Quotas: one item per image; per-
target cap of 5 within each bin, relaxed stepwise only if a full sweep cannot fill
the bin.

## Output and realized statistics (v1)

`f1_S_candidates_v1.jsonl` — 500 items, fields: `id, family, prompt, target, cue,
template_id, plausibility, scenario, negbench_image_id, caption_index, source`.

From `f1_S_candidates_v1.stats.json`: 250/250 per bin; 75 distinct targets (max 11
items per target); 500 distinct images; mean prompt length 15.1 words; templates
T1–T5 = 107/105/86/113/89; cues no/without = 306/194.

## Human pass

The 500 are candidates, not the released set. A human pass trims to the final ~200,
removing: COCO caption defects (typos such as "eachother", "coach" for "couch"),
awkward attachment of the negation phrase, and borderline implicit mentions the
string check cannot see (e.g. a scene whose description implies the target). All six
families receive the same final human pass, so construction method differs across
families only in where candidates come from — this is what licenses comparing
hand-collected and rule-built families in one table.

## Known limitations

- Target vocabulary is COCO-80 (inherited). Detector-friendly by construction
  (presence judging can use any COCO-trained or open-vocabulary detector), but
  object-centric: no scene-level or abstract targets.
- Scene descriptions are COCO val2017 captions: everyday photographic scenes,
  photographic register, occasional annotator typos.
- The implausible bin's zero-co-occurrence criterion is computed on val2017 positives
  (5,000 images); it is a corpus estimate of incongruity, not a semantic guarantee.
