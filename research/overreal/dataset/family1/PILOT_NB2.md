# Family 1 x Nano Banana 2 — generation run and full annotation

## Generation

| run | items | samples | images | API | outcome |
|---|---|---|---|---|---|
| `nb2_pilot` | 20 stratified | 2 | 40 requested | standard | 31 ok, 9 failed on a 429 monthly-spending-cap error (billing, not safety) |
| `nb2_full` | all 500 | 2 | 969 requested (the 31 pilot successes excluded) | batch | 969/969 ok |

Config identical across both: `gemini-3.1-flash-image`, 1K, 1:1,
`thinking_level=minimal`, deterministic per-(item, sample) seed recorded in the
manifest. **Coverage: 1000/1000 (item, sample) pairs**, no safety blocks, no
text-only responses. Batch cost ~$32.

## Annotation

All 1000 images were annotated. Protocol: each image viewed once at normal scale by
a model annotator (Claude subagents, 20 x 50 images); **no cropping, zooming or
magnification** — anything not identifiable at normal viewing scale is recorded
`realized=false` with `confidence="low"`. `realized=true` means the target object
appears, i.e. an over-realization, since every prompt asked for its absence.

Per-image labels with evidence sentences: `annotations_nb2.jsonl`
(fields: key, item_id, sample, target, plausibility, prompt, realized, confidence,
depiction_only, evidence). Summary: `annotations_nb2.stats.json`. Raw per-chunk
output kept in `images/inspect_chunks/`. Builder:
`scripts/aggregate_family1_inspection.py`.

Standing as evidence: one rater, and a model rater. Adequate for difficulty triage
and for characterising failure modes (EXPERIMENT.md §1.3); not ground truth for a
published rate without a second annotator.

## Result: 25/1000 over-realized (2.5%)

| set | n | realized | rate | 95% CI (Wilson) |
|---|---|---|---|---|
| all | 1000 | 25 | **2.5%** | 1.7–3.7% |
| plausible bin | 500 | 22 | **4.4%** | 2.9–6.6% |
| implausible bin | 500 | 3 | **0.6%** | 0.2–1.8% |

**The plausibility factor separates: OR = 7.6, Fisher exact p = 1.3e-4.** A target
that belongs in the scene is roughly seven times more likely to survive an explicit
negation than one that does not. This **corrects the reading in the earlier
60-image sample**, which found 1/60 and concluded both bins were at ceiling; at that
sample size the bin difference was invisible.

Per item (either sample counts): 22/500 items over-realized at least once, and only
3/500 on both samples — so most failures are sampling-level, not item-level. That
matters for item selection: a single generation per item would misclassify most of
these items.

Sensitivity: restricting to `confidence="high"` annotations gives 6/1000 (0.6%);
excluding the two depiction-only cases gives 23/1000. The headline 2.5% is therefore
an upper-ish estimate that includes medium-confidence background objects.

### Where the failures concentrate

By target: keyboard 4, spoon 3, cup 3, backpack 2, dining table 2, then eleven
classes with one each. These are **small, high-prior scene furniture** — objects a
photograph of that scene would ordinarily contain. No large or focal target
(elephant, bus, train) was ever realized.

### Two judgement boundaries the paper must state

1. **Depiction-only** (2 cases). `f1_nb_340_s1`: sheep appear on a TV screen inside
   the scene. `f1_nb_426_s1`: a traffic light appears in a framed photograph on the
   wall. Whether a depicted target counts as realization is a decision, not an
   observation; both are flagged `depiction_only=true` so either policy can be
   applied post hoc.
2. **Class semantics.** `f1_nb_134_s0/s1` (keyboard): the only keyboard is the
   laptop's own integrated one — if the intended reading is "no separate keyboard
   peripheral", these flip to false. `f1_nb_304_s0/s1` (cup): a plain water tumbler
   counted as COCO `cup`. Fixing a written class-semantics rule before the
   multi-annotator pass will remove most inter-rater noise.

## Suppression is visible, and it has side effects

The most interesting material is not the failure count but *how* NB2 suppresses. On
`keyboard` items (16 images, 4 realized) three distinct strategies appear:

- **Erasure**: `f1_nb_136_s0/s1`, `f1_nb_276_s1`, `f1_nb_496_s1` — an open laptop is
  rendered with a **blank, keyless base**; one keeps only the Apple logo where the
  key deck should be.
- **State change**: `f1_nb_477_s0` — the laptop is simply closed.
- **Reframing**: `f1_nb_477_s1` — the crop stops above the key deck.

And on `dining table`: `f1_nb_286_s1` replaces the table with a **wine barrel**
carrying the cake, while its paired sample `f1_nb_286_s0` renders an ordinary table
and fails.

This is a finding in its own right: the model does not merely omit, it **repairs the
scene**, sometimes at the cost of physical plausibility (a keyless laptop). It also
suggests a measurement the benchmark could add — a suppression *cost*, distinct from
suppression *success*.

## Open items

- **Automatic judge over all 1000.** `scripts/judge_family1_batch.py` implements the
  corrected binary protocol (Qwen2.5-VL-7B). Not run: the CUDA driver fails to
  initialize in the agent environment (`cuInit` -> initialization error). Run with
  `CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/judge_family1_batch.py --dir
  dataset/family1/images/nb2_full`.
- **Judge certification.** `annotations_nb2.jsonl` is the reference set for the
  (Qwen, binary question, NB2) triple required by EXPERIMENT.md constraint 3. With
  25 positives in 1000 there is now enough signal for a meaningful kappa.
- **One known annotator disagreement.** `f1_nb_216_s0` (kite through a cafe window):
  scored absent in the earlier 60-image pass, present in the full pass. Recheck when
  the second annotator runs.
- **Seed honoring.** Still untested.
