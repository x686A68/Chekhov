# R16: attention intervention

Design 2026-09-25. Reviews A, C#5 and the fourth review ask whether the
attention pattern of 6.3 is causally connected to over-realization. We add a
logit bias log(k) to the image-to-text attention scores of chosen text
tokens (all blocks, all steps, conditional branch only) and regenerate.

## Conditions (intervene_common.py)
- cue_x2/4/8   S: cue words (the words the plain-mention control replaces) x k
- tgt_d2/4/8   S: target words / k
- rand_x8      S: one content word outside target and cue, x 8 (magnitude control)
- P_rep_x8     P: the replacing words of the control x 8 (specificity: the
               requested target should stay)
- P_tgt_d8     P: target / 8 (erasure check: does /8 remove a requested target too)
Cue words are the word-level diff between S and P (attn_common.diff_spans),
not the c_spans_* fields of pairs.jsonl (those are the probe's control word).

## Data
- SD3.5-Large: the 328 S/P items of pairs.jsonl that have an Opus baseline
  label in the Table 1 sample (intervene_items.json), seeds 0 and 1, all
  nine conditions; steps and cfg from the benchmark manifest.
- FLUX.1-dev and Qwen-Image: 40 items per family (intervene_items_small.json),
  seed 0, conditions cue_x8, tgt_d8, rand_x8, P_rep_x8.

## Judging
No API. intervene_judge.py asks Qwen2.5-VL-7B whether the target is visible
(readable, for text targets), for every intervention image and the matching
baseline (S: benchmark image; P: attn/<model>/*__P_s*.png). Presence, not
the four-way label; the silent/integrated split is checked by hand on
contact sheets (intervene_sheets.py). intervene_analyze.py reports presence
per condition and family with the paired difference to the baseline and a
bootstrap over items.

## Predictions written before the run
1. cue_x8 lowers target presence on existence-canceling, mental-state and
   figurative prompts; P_rep_x8 leaves the requested target in place.
2. cue_x8 does nothing on perspectival prompts (the encoder has already lost
   the information, D.7).
3. rand_x8 does nothing.
4. tgt_d8 removes the target from S and P alike (erasure, not withholding).

## Results (2026-09-26; numbers in results/intervention_summary.md)

Judge: local Qwen2.5-VL-7B, "is the target visible" (readable, for text
targets). On the baseline images it agrees with the Opus label collapsed to
present (DO/SO/I) vs absent (W) on 91% (SD3.5-L, 455 images; perspectival
81%, the other families 90-96%), 97% (FLUX) and 93% (Qwen-Image).

SD3.5-Large, 328 items x 2 seeds, target presence rate and paired change:

| condition        | EC        | MS        | Fig       | Per       | all       |
|------------------|-----------|-----------|-----------|-----------|-----------|
| baseline         | .62       | .85       | .71       | .79       | .75       |
| cue x2 / x4 / x8 | .00/.01/.00 | +.04/-.01/-.01 | +.01/-.02/-.03 | +.02/.00/+.02 | +.02/-.01/-.01 |
| cue x8, all rows | -.09*     | -.04      | -.05*     | -.01      | -.04*     |
| target /2 /4 /8  | -.25/-.37/-.46* | -.06/-.22/-.37* | -.18/-.31/-.38* | -.05/-.13/-.25* | -.12/-.25/-.35* |
| random word x8   | -.09*     | -.01      | -.09*     | -.02      | -.05*     |
| function word x8 | -.02      | +.01      | -.05      | .00       | -.02      |
| P: replacing x8  | +.07*     | +.02      | .00       | +.02      | +.02*     |
| P: target /8     | -.52*     | -.18*     | -.19*     | -.26*     | -.26*     |

FLUX.1-dev (160 items, seed 0): cue x8 +.01, target /8 -.24* (EC -.20*, MS
-.07, Fig -.40*, Per -.30*), random x8 -.03, P replacing x8 +.01.
Qwen-Image (80 items, seed 0): cue x8 -.05, target /8 -.20* (EC -.35*, MS
.00, Fig -.30*, Per -.15), random x8 .00, P replacing x8 -.03.
P: target /8 on the control: FLUX -.17* (EC -.28, MS -.12, Fig -.15, Per
-.12), Qwen-Image -.11* (EC -.15, MS .00, Fig -.20, Per -.10). The
figurative asymmetry (mention removed more easily than request) holds in all
three models (SD3.5 -.38 vs -.19, FLUX -.40 vs -.15, Qwen -.30 vs -.20); the
mental-state asymmetry only in SD3.5.

### Reading

1. Raising the attention on the cue words does not lower the target's
   presence: x2, x4, x8 on the image-query rows change nothing in any
   family or model. Prediction 1 fails. Biasing the text-query rows too
   (cue_x8_all) gives -.04 overall, the same size as the random content
   word (-.05), while the function-word control moves nothing (-.02), so it
   is a content-word competition effect, not a cue effect.
2. Lowering the attention on the target words lowers its presence with a
   monotone dose-response in every family and model, but /8 still leaves
   the target in 40-55% of the images; perspectival is the most resistant
   (-.25), Qwen-Image's mental-state prompts do not move at all.
3. The mentioned target is easier to remove than the requested one on
   mental-state and figurative prompts (S -.37/-.38 vs P -.18/-.19 at /8),
   not on existence-canceling (-.46 vs -.52) or perspectival (-.25 vs -.26).
   This matches D.7/D.8: where the encoder separates the two prompts and
   the backbone's rho_t is lowest, the backbone also holds the mention more
   loosely.
4. Specificity holds: x8 on the replacing words of the control leaves the
   requested target in place (+.02).
5. The random content word is not a neutral control: raising a scene word
   competes with the target for the composition (-.09 on EC and figurative,
   visible on the contact sheets). The function-word control (func_x8) is
   the neutral one.

### Consequence for the paper

The attention deficit on the cue words is a symptom, not a lever: the
information is present (D.7) and the backbone can be made to read the cue
words eight times more without changing what it draws. Attention on the
target word is a lever for presence, as expected, and removing a mention
takes less than removing a request on the two families where the encoder
carries the distinction. This does not restore "mechanism" for 6.3; it
sharpens the 6.3/D.7 picture and supports the two-regime reading (R18).

Recommended model to report: SD3.5-Large (full design: three doses, all-rows
and function-word controls, both sides of the pair, 328 items x 2 seeds).
FLUX.1-dev replicates points 1, 2 and 4 on 160 items; Qwen-Image on 80
items, with the mental-state anomaly noted.
