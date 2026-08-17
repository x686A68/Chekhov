# Family 1 x Nano Banana 2 — generation run and hand inspection

## Runs

| run | items | samples | images | API | outcome |
|---|---|---|---|---|---|
| `nb2_pilot` | 20 stratified | 2 | 40 requested | standard | 31 ok, 9 failed on a 429 monthly-spending-cap error (billing, not safety) |
| `nb2_full` | all 500 | 2 | 969 requested (the 31 pilot successes excluded) | batch | **969/969 ok** — no safety blocks, no text-only responses, no errors |

Config identical across both: `gemini-3.1-flash-image`, 1K, 1:1,
`thinking_level=minimal`, deterministic per-(item, sample) seed recorded in the
manifest. Total coverage: 1000/1000 (item, sample) pairs. Batch cost ~$32.

## Hand inspection (60 images, stratified)

Sample: 30 plausible / 30 implausible drawn with seed 31415 from `nb2_full`
(`inspect_sample.jsonl`); labels in `inspection.jsonl`. One rater (Claude), presence
of the target in any depiction. Per EXPERIMENT.md §1.3 this is difficulty triage and
a judge-certification reference, **not** a publishable rate.

**Result: 1/60 over-realized.**

- `f1_nb_392_s0` (plausible, *backpack*): "the train is crossing a street in the
  city, without any backpack" — at least two pedestrians wear black backpacks (one
  in the left foreground with a green tag on the strap, one in the right-hand group).
  Both are small and peripheral; the failure is in incidental background detail, not
  in the scene's subject.

The other 59 suppressed. Cases worth recording because they show the discrimination
is real rather than accidental:

- `f1_nb_235_s0` (*spoon*): plate of food rendered **with fork and knife, no spoon**.
- `f1_nb_360_s0` (*truck*): busy London street with buses, taxis and cars, **no truck**.
- `f1_nb_183_s1` (*mouse*, pilot): workstation with monitor and keyboard, **no mouse**,
  twice.
- `f1_nb_285_s1` (*baseball glove*): dog sniffing a bat on a field, **no glove**.

Borderline calls, all resolved as suppressed, recorded in `inspection.jsonl`:
`f1_nb_168_s0` (cafe sign shows fried pastries, not clearly donuts), `f1_nb_475_s1`
(blurred background players in pitching motion, no bat visible), `f1_nb_263_s0` (cut
flowers in a vase, not a potted plant), `f1_nb_362_s1` (built-in microwave present;
COCO treats microwave and oven as distinct classes).

## Reading

1. **NB2 is at or near ceiling on explicit trailing negation** over COCO-80 objects.
   Combined with the pilot's 0/31 on non-overlapping images: 1 realization in 91
   hand-inspected images (~1.1%, rule-of-three 95% upper bound ~4% given zero-ish
   counts). The one failure is a small peripheral object in a crowded street scene,
   which suggests the residual failure mode is **incidental scene furniture**, not
   the prompt's focal content.
2. Consistent with EXPERIMENT.md §2.3: a commercial model that rarely over-realizes
   is a result to report, not a hole to patch. Cross-condition pairing still yields
   certain Det positives from it.
3. Both plausibility bins are at ceiling, so the bin factor buys nothing **for this
   generator**. It may still separate weaker generators (the probers), which is where
   Family 1's difficulty headroom has to come from.
4. Item difficulty for OverReal-Gen must therefore be established with the probers
   (SDXL / SD3.5-Large), not with NB2.

## Not yet done

- **Automatic judge over all 969.** `scripts/judge_family1_batch.py` implements the
  corrected binary protocol (Qwen2.5-VL-7B). It could not be run from the agent
  environment — the CUDA driver fails to initialize there (`cuInit` -> initialization
  error) although the device nodes are visible.
- **Judge certification for this generator.** EXPERIMENT.md constraint 3 requires
  re-certifying the (model, question, generator) triple. `inspection.jsonl` is the
  reference set: once the judge has run, compute kappa on these 60. With only one
  positive, kappa will be unstable — treat a disagreement on `f1_nb_392_s0` as the
  informative event, and expand the inspected set if the judge flags positives the
  rater did not see.
- **Seed honoring.** Untested. Rerun one (item, sample) with its recorded seed and
  compare compositions.
