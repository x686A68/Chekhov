# Family 1 x Nano Banana 2 — pilot readout (2026-08-15)

Run: `run_images_nanobanana.py` defaults — 20 stratified items (10 plausible / 10
implausible) x 2 samples, `gemini-3.1-flash-image`, 1K, 1:1, thinking=minimal,
standard API. Inspection: single rater (Claude), presence-of-target only; this is
difficulty triage in the sense of EXPERIMENT.md §1.3, not a publishable rate.

## Completion

31/40 images generated. 9 calls failed with 429 "monthly spending cap exceeded"
(billing, not safety) — items f1_nb_380(s1), 396, 414, 429, 444 missing. No safety
blocks, no text-only responses. Mean latency 14.5 s. Rerun after raising the cap at
ai.studio/spend; the runner resumes from the manifest.

## Result: 0/31 over-realized

Every generated image correctly suppressed its target, in both bins. Notable cases:

- `f1_nb_183` (plausible, *mouse*): a computer workstation rendered twice with
  keyboard and monitor and **no mouse** — suppression against a strong co-occurrence
  prior, twice.
- `f1_nb_301` (implausible, *cow*): no cow; incidentally the desk contains a computer
  mouse — coincidental realization of *other* items' targets is alive and well, which
  is what the A-condition base rate is for.
- `f1_nb_216` s0 (*kite*): small triangular pennant outside the window — nearest
  thing to a borderline case in the batch; judged not-a-kite.
- Fidelity errors unrelated to the construct occur (e.g. "a shoe" rendered as a
  pair) — irrelevant to presence judging.

With n=31 and zero events, the rule-of-three 95% upper bound is ~9.7%: this pilot
says the rate is likely below ~10%, not that it is zero.

## Implications

1. **NB2 appears at/near ceiling on explicit trailing negation** over
   detector-friendly COCO objects. Consistent with EXPERIMENT.md §2.3: a commercial
   model that rarely over-realizes is a result to report, not a hole to patch —
   cross-condition pairing still yields certain Det positives from it.
2. For OverReal-Gen difficulty, Family 1 explicit items will likely need the
   **probers** (SDXL/SD3.5-Large), not NB2, to discriminate; and the implicit
   variants / harder constructions carry the family's headroom.
3. Seed honoring still untested: s0/s1 used different seeds by design. Test = rerun
   one (item, sample) with its manifest seed and compare compositions.

## Costs

31 images ≈ $2.1 at the 1K standard rate.
