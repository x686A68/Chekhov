# 6.3 "Text encoder": does the encoder represent the space builder?

Design for the first paragraph of paper section 6.3 (The mechanism behind
over-realization). Written 2026-09-15. The other two paragraphs (Diffusion
backbone, Causal analysis) get their own files when they start.

## Question

Section 3.2 claims the model represents a mention as content to depict. The
first place that could happen is the text encoder: the diffusion backbone never
sees the prompt, only the encoder's output. So the question here is whether the
representation the backbone receives still distinguishes "the target is under a
space builder" from "the target is plainly requested". If it does not, the
failure starts in the encoder. If it does, the encoder is not the bottleneck and
the loss happens in the backbone (paragraphs 2 and 3).

## Prompt conditions

Three versions of every OverReal-Gen item, same target word in all three:

| code | name in paper | status | example (figurative) |
|---|---|---|---|
| S | space-builder prompt | exists (`data/generation/prompts.jsonl`) | The coach is as fierce as a tiger. |
| P | plain-mention control | **to write** | The coach is standing next to a tiger. |
| A | target blank | exists (`data/generation/notarget_placeholder_prompts.opus.jsonl`) | The coach is as fierce as someone. |

Only S and P are used by the probe. A is kept as an optional third class for a
sanity check (a probe that cannot tell A from S has a broken feature pipeline).

### Writing P

- One rewrite per item, **written by hand (no API calls; author rule
  2026-09-15)**, minimal edit: replace the space-builder construction with a
  plain presence construction, keep the target string verbatim, keep
  everything else, length within 3 words of S.
- Family recipes given in the instruction: cancellation "no X / without X" ->
  "an X / with an X"; mental-state "remembering / believing there is X" -> "next
  to X / with X in front of her"; figurative "as ADJ as X / like X" -> "next to
  X / with X beside him"; perspectival "X hidden behind / inside / facing away"
  -> "X in front of / on top of / facing the camera".
- Automatic checks: target present verbatim; word count within tolerance.
- Family 1 is run with the 50 cancellation items that exist in the generation
  data (author decision 2026-09-15).

### Target spans

The probe reads the target's token span, so every S and P needs a character
span for the target. 29 of the 548 non-cancellation targets are not verbatim in
the prompt (typos such as "birthday cale", "hot potatoe"; paraphrases such as
"eggs basket" for "all their eggs in one basket"). Fix by adding a `target_span`
field: automatic when the target is verbatim, hand-written for the 29. The
target field itself is not changed.

## Encoders

Exactly the encoders the evaluated pipelines use, loaded from the pipeline
repos so that weights, tokenizer and preprocessing match generation:

| encoder | used by | what the backbone receives (diffusers) | what we probe |
|---|---|---|---|
| CLIP-L/14 | SD3.5 (seq + pooled), FLUX (pooled only) | SD3: `hidden_states[-2]`; FLUX: `pooler_output` | target span of `hidden_states[-2]`; pooled vector |
| CLIP-bigG/14 | SD3.5 (seq + pooled) | `hidden_states[-2]` | target span; pooled |
| T5-XXL | SD3.5 (256 tok), FLUX (512 tok) | `last_hidden_state` | target span; mean over prompt |
| Qwen2.5-VL-7B | Qwen-Image | `hidden_states[-1]` after the describe-the-image system template, first 34 template tokens dropped | target span; mean over prompt |

Notes:
- FLUX gets nothing from CLIP but the pooled vector, so for FLUX the CLIP
  question is whether the pooled vector separates S from P.
- Qwen-Image wraps the prompt in a system instruction "Describe the image by
  detailing the color, shape, ...". We encode with that template, since that is
  what the backbone sees.
- A layer sweep (every layer of each encoder) goes to the appendix. The main
  table reports the deployed layer only.

## Representations and probes

For each encoder and each item, two feature vectors:

1. **target span**: mean of the contextual token vectors covering the target
   (subword pieces averaged). This is the main probe: does the target's own
   representation know it is under a space builder?
2. **pooled / mean**: the pooled vector (CLIP) or mean over prompt tokens (T5,
   Qwen). Secondary; it is what FLUX gets from CLIP.

Probe: logistic regression on standardized features, S vs P, chance 50%.
Regularization chosen by inner cross-validation.

Cross-validation: **GroupKFold with the target lemma as the group**, 5 folds,
so no target word appears in both train and test. Otherwise the probe can
memorize which targets tend to come with which construction.

Training regime: **one probe per encoder, trained on all four families
together** (decided 2026-09-15). The label means the same thing in every
family, "this mention is not a request", and the paper treats the four
families as one phenomenon, so the probe should too. Families are weighted so
that each contributes equally to the loss. Test accuracy is still reported
**per family** (each family's held-out folds), plus pooled.

No per-family probes and no leave-one-family-out setting: the pooled probe
with per-family test accuracy is the whole result (author decision
2026-09-15).

Family sizes (S+P): cancellation 2x50 as generated (`prompts.jsonl` and every
manifest hold 50 cancellation items, while Table 1 of the paper says 200; to
be reconciled before the run), mental-state 2x75, figurative 2x237,
perspectival 2x186; pooled 1,096 to 1,396. Report bootstrap 95% intervals.

Small-sample safeguards: regularization chosen inside the training folds
only; a label-shuffle null distribution reported next to every accuracy; and a
training-free mass-mean probe (difference of class means as the direction)
reported alongside logistic regression.

## Controls

1. **Other-word probe.** Train the same S-vs-P probe on the span of a content
   word that is not the target and not part of the space builder (the scene
   noun nearest the target, chosen automatically and spot-checked). If it also
   separates S from P, the separability is prompt-wide style, not knowledge
   about the target. The claim needs target-span accuracy clearly above
   other-word accuracy.
2. **Cue-word hold-out.** Second cross-validation that groups by the cue word
   (no / without / never; believes / remembers / ...; like / as / -like;
   behind / inside / ...). Tests whether the probe learns the relation or a
   list of cue tokens. Reported in the appendix.
3. **Length and wording match.** P rewrites are minimal edits; report the mean
   length difference S vs P per family. If a family's P prompts are
   systematically longer, note it.

## Bridge to the generation results

Two analyses that connect the encoder to Table 1:

1. **Item-level margin vs over-realization.** For each S item, the probe's
   held-out decision margin (how confidently the encoder marks the target as
   space-builder-bound). Correlate with the item's over-realization rate across
   seeds for the model that uses that encoder (SD3.5-L, FLUX, Qwen-Image; rates
   from the Opus auto-annotation). A near-zero correlation with a high probe
   accuracy is the strongest form of "the encoder knows, the backbone does not
   use it".
2. **Ask the encoder directly (Qwen only).** Qwen2.5-VL-7B is an instruction
   model. Ask it for S and P: "Is a <target> physically present in the scene
   this prompt describes? Answer yes or no." Accuracy per family. This is the
   cheap replacement for the deleted text-control section: the same weights
   that encode the prompt for Qwen-Image answer the question correctly when
   asked, and still condition an image that over-realizes.

## Expected pattern

- CLIP-L and CLIP-bigG: near chance on cancellation and mental-state, weak on
  figurative and perspectival (consistent with the negation results in
  NegBench and related work).
- T5-XXL and Qwen2.5-VL: high on all four families.
- Other-word control well below the target-span probe.
- Margin vs over-realization: weak or absent.

If T5 and Qwen probes are high, the paragraph concludes that the space builder
survives encoding for every model except through the CLIP channel, and the
next paragraph asks how the backbone reads it. If instead T5 or Qwen are near
chance for some family, that family's failure is located in the encoder and
paragraph 2 should treat it separately.

## Outputs

- `data/generation/plain_prompts.opus.jsonl`: P prompts with `target_span`.
- `data/generation/prompts.jsonl` gains `target_span` (S).
- `research/overreal/mechanism/features/<encoder>.npz`: per-item features,
  all layers.
- `research/overreal/mechanism/results/text_encoder.json` and a table script
  that prints the LaTeX rows for the paper.

## Cost

Encoder forward passes over 1,396 prompts: minutes on one GPU. P rewriting:
698 Opus calls plus the check calls. Manual review: about 260 prompts
(families 2 and 4) plus a sample of 90.

## Environment

`~/miniconda3/envs/Fraud/bin/python` (diffusers 0.40, transformers 5.8,
scikit-learn 1.6). Model weights in `/data/users/jiahao_huang/hf`
(`HF_HOME`). All four encoders are already cached inside the SD3.5, FLUX and
Qwen-Image repos.

## Results (run 2026-09-15)

Files: `results/text_encoder_<encoder>.json`, `results/layer_sweep_<encoder>.json`,
`results/layer_sweep.pdf`, `results/ask_encoder.jsonl`, `results/text_encoder_table.tex`.
Probe settings: logistic regression, C from {0.01, 0.1, 1, 10} by inner CV
(0.01 chosen almost everywhere), 5 GroupKFold folds by target word, family
weights, 1000 bootstrap resamples, 50 label-shuffle nulls (95th percentile
0.51 to 0.54).

Target-word probe at the deployed layer (EC / MS / Fig / Per / pooled):

| encoder | target word | nearest other word |
|---|---|---|
| CLIP-L/14 | .95 .81 .87 .53 / .79 | .50 .81 .63 .54 / .62 |
| CLIP-bigG/14 | .96 .87 .87 .60 / .82 | .50 .77 .65 .61 / .63 |
| T5-XXL | .95 .89 .94 .62 / .85 | .78 .82 .81 .66 / .77 |
| Qwen2.5-VL-7B | 1.00 .92 .91 .60 / .86 | .50 .75 .69 .63 / .64 |

Prompt-mean probe: .84 / .86 / .92 / .91 pooled (CLIP-L, bigG, T5, Qwen);
perspectival .77 / .80 / .83 / .84. CLIP-L pooled vector (FLUX): .80 .79 .82
.67 / .77.

Layer sweep: perspectival stays in the chance band at every layer of every
encoder; the other three families separate within the first 4 CLIP layers
and rise gradually in T5 and Qwen.

Bridge (Spearman rho, probe margin vs item over-realization rate of the
generator using that encoder, raw prompts, Opus labels): within-family rho
between -0.28 and 0.28, no consistent sign; pooled rho slightly negative
(-0.15 to -0.20 for T5 and Qwen) which is the perspectival family confound
(low margins, high rates), not a within-family effect.

Ask the encoder (Qwen2.5-VL-7B, "would the target be visible", S expects no,
P expects yes): EC 1.00 / .76, MS .75 / 1.00, Fig .84 / .93, Per .56 / .85.
Perspectival errors: "closed container, inside is X" is answered yes.
Cancellation P errors: implausible scenes (a toilet on a tray) answered no.

Reading for the paper: for EC, MS and Fig the space builder survives
encoding on the target word itself, and the encoder's confidence does not
predict the generator's failure, so the loss is downstream (paragraphs 2 and
3). For Per the target's representation never records that it is out of
view, in any encoder and at any layer, so that family's failure begins in
the encoder.

Caveats: the mental-state control word is often the verb the rewrite
replaces, so its control accuracy is inflated; the T5 control word is high
in general (T5 mixes context into neighbours strongly). Family 1 has 50
items only.
