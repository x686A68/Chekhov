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
