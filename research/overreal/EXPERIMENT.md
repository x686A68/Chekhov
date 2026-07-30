# EXPERIMENT.md — OverReal experimental design

The core design for the two-part benchmark. `GOAL.md` was the pilot spec and is now
history; `DECISIONS.md` holds the rationale and the rejected alternatives; `pilot/REPORT.md`
holds the evidence this design rests on. This file is what to execute.

**Scope (DECISIONS.md #36–39).** Text-to-image only. Two tasks:

- **OverReal-Gen** — does the model suppress content the prompt does not license?
- **OverReal-Det** — given the prompt, can a model tell that an image over-realized?

Text is retained only as a ~36-item-per-family control that establishes the modality
asymmetry. It is not part of either task.

---

## 0. The four constraints that shape everything below

Each was learned the expensive way in the pilot. They are listed first because most of
the design decisions are consequences of them.

1. **The prober set and the evaluated set must be disjoint.** Selecting items because a
   model fails them and then reporting that model's failure rate on those items measures
   the selection, not the model. (DECISIONS.md #21, violated by the obvious reading of
   Phase 1.)
2. **Difficulty filtering destroys the base rate.** If only failing items survive, the
   headline rate is ~1.0 by construction. The unfiltered pool is the primary rate; the
   filtered set ranks models. Both are released.
3. **Judge reliability is a property of the (model, question, generator) triple.** One 7B
   VLM scored identical images at κ=0.23 and κ=0.94 on wording alone (REPORT §4.2–4.4).
   Every new generator requires re-certification, because it can produce failure modes the
   question does not cover.
4. **Detection labels must not come from a judge, and must not come from the generating
   condition.** Cross-condition pairing solves both (DECISIONS.md #40).

---

## 1. Phase 1 — build OverReal-Gen

### 1.1 Item pool

Per family: 60 items. Entity-to-scenario ratio is family-specific (DECISIONS.md #25):
relevance varies conversations (20 × 3 entities), 4b varies carriers (20 × 3 contents),
the object families vary entities against scenarios by Latin square. Naturalness is
human-rated on a sample and reported, never used as an automatic filter (#35).

Conditions per item: `S_imp`, `S_exp`, `P`, `A`. Openness levels: L1 fully specified,
L2 conventional, L3 sparse. So 12 prompts per item, **4,320 prompts per family set**.

### 1.2 The difficulty loop, with the prober/eval split

```
candidates → PROBER generators → my inspection → keep / revise / drop → OverReal-Gen
                (SDXL, SD3.5-Large)                                          ↓
                                                    EVAL generators never see this loop
                                                    (FLUX.1-dev, Qwen-Image, SD3.5-Medium)
```

- **Probers** decide difficulty. **Eval generators** are never used to select items.
- An item that no prober fails is **revised**, not deleted, on the first pass; deleted only
  if two revisions fail to make it discriminating. Record every revision — GOAL.md rule 4
  and the pilot's own history (three constructions of family 5, three of 4a) show that the
  discarded versions are evidence.
- **Any revised item re-enters the A condition check.** Revision is exactly when base-rate
  leaks appear: the pilot's family-6 neutral context said "wildlife park" and pulled
  animals into A. Non-negotiable.
- Release **both** the pre-filter pool and the post-filter set.

### 1.3 My role, and its limit

I inspect prober output item by item and classify: over-realized / suppressed / oblique /
ambiguous. This is one rater. It is adequate for *difficulty triage* — deciding whether an
item discriminates — and it is **not** adequate as ground truth for any published rate.
Everything reported in Phase 3 is either automatic-with-human-certification or
multi-annotator. Where my judgement is the only source, it is labelled as such.

---

## 2. Phase 2 — build OverReal-Det

### 2.1 Where the items come from

Run OverReal-Gen prompts through **commercial** T2I models. This is the one place the
"no external APIs" constraint is relaxed, and only for *building* the dataset — nothing in
the evaluated pipeline calls an API. State this explicitly in the paper.

### 2.2 Item construction — three classes, two of them free

Using the shared seed that makes S, P and A near-identical in composition:

| item | label | source of the label | annotation cost |
|---|---|---|---|
| `S` prompt + `P` image | over-realizes | **construction** — image generated from an explicit request for E | none |
| `S` prompt + `A` image | does not | **construction** — licensed by the measured A base rate | none |
| `S` prompt + `S` image | either | **human annotation** | the whole cost |

The first two are the **construction-guaranteed core**; the third is the **natural set**.
They are reported separately and never averaged: the core asks whether a detector can
relate a prompt to an image at all, the natural set asks whether it can do so on the
distribution a generator actually produces.

### 2.3 Drop the adversarial fallback

The proposed fallback — hand-write attack prompts if commercial models rarely fail — is
**removed from the design**. It would put positives and negatives on different
distributions (hand-crafted vs naturally generated), letting a detector win on
distribution shift rather than on the licensing relation. It is also unnecessary:
cross-condition pairing yields certain positives from *any* generator, however well it
behaves. If commercial models turn out never to over-realize, that is a result to report,
not a hole to patch.

### 2.4 Controls — publication conditions, not robustness checks

Every detector is run three times:

1. normal (prompt + image)
2. **prompt withheld** (image only)
3. **prompt mismatched** (image + a different item's S prompt)

Above-chance performance in (2) or (3) means the cell leaks and is **printed as void**,
not quietly dropped. Additional requirements: candidates come from ≥2 generators; no
evaluated detector may be a model used anywhere in constructing labels; a detector is never
scored on images produced by itself (self-evaluation bias) — report with and without if
unavoidable.

---

## 3. Phase 3 — the two main tables

### 3.1 Table A: local generators on OverReal-Gen

**The cost problem and its solution.** 4,320 prompts × 3 eval generators = 12,960 images.
Human-annotating that is impossible. The pilot licenses the alternative: the corrected
binary judge reaches κ = 0.94–1.00 against hand annotation, including 0.955 and 1.00 on two
families it was never tuned on (REPORT §4.4).

So: **automatic judge over the full grid, human annotation on a stratified certification
subset.**

| | count | note |
|---|---|---|
| judged automatically | 12,960 | ~2.5 GPU-hours at 0.67 s/image |
| human-certified subset | 300 per (generator × family) — 5,400 total | stratified over condition and openness |
| re-certification trigger | any new generator, any question rewording | κ is a property of the triple (#0.3) |

Report κ per (generator × family) beside the rate. A family whose κ falls below a
pre-registered threshold (propose 0.80) is reported as human-only, not as a rate.

**Generation cost:** 12.1 s/image measured for FLUX at 1024², 50 steps. 12,960 images ≈
44 GPU-hours on one H100, ~22 on two. Qwen-Image is larger and will be slower; measure
before committing. 28 steps roughly halves it and is worth piloting first.

### 3.2 Table B: local detectors on OverReal-Det

Reported as a grid of **detector × family × item class**, with the two control columns
beside every cell. Never a single pooled accuracy.

---

## 4. Phase 4 — analysis

### 4.1 The structural problem with "gen–det correlation"

The two tasks use **disjoint model classes**: diffusion generators cannot judge, and VLM
detectors cannot generate. There is no model with a score in both tables, so a per-model
correlation is not computable for the main roster. Two analyses instead:

**(a) Family-level correlation — the main result.** Are the families that generators fail
most the families detectors miss most? Computable for every model, and it answers whether
the difficulty of producing and the difficulty of recognising track each other.

**(b) Unified models — the only per-model answer.** Janus-Pro-7B, BAGEL-7B-MoT and Emu3
both generate and judge. On those, and only those, ask directly: *does a model that makes
this error recognise it?* This deserves its own subsection because it is the only place the
paper can answer that question, and it is the most interesting thing in Phase 4. Expect
engineering friction — these have far fewer downloads than the mainstream models.

### 4.2 What must not be claimed

- No representational mechanism. The availability-vs-expression analysis went with the text
  half (DECISIONS.md #38).
- No cross-modal openness comparison. Openness levels are constructed, not calibrated.
- Nothing about proprietary generators' failure *rates* — commercial models appear only as
  a source of Det items, on a non-random prompt set.

---

## 5. Model roster

| role | models | why |
|---|---|---|
| **prober** (selects items) | SDXL, SD3.5-Large | different lineage from the eval set; SDXL is weak enough to filter out easy items |
| **eval generator** | FLUX.1-dev, Qwen-Image, SD3.5-Medium | spans architectures; Qwen-Image is the only realistic hope for family 5, whose failure did not reproduce on FLUX because FLUX writes pseudo-text rather than real words (REPORT §3.1) |
| **detector** | Qwen2.5-VL 7B / 32B / 72B | a **size ladder inside one family** separates scale from family — more informative than several different 7Bs |
| | InternVL3 8B / 38B | second family, also laddered, for cross-family validation |
| **unified** | Janus-Pro-7B, BAGEL-7B-MoT | the only models that can appear in both tables (§4.1b) |

All verified present on HF. `FLUX.1-dev`, `SD3.5-Large` and `SD3.5-Medium` are
`gated=auto` and need a licence click; everything else is ungated.
`Llama-3.2-Vision` is `gated=manual` and is deliberately excluded — approval latency is a
schedule risk for no unique capability.

---

### 5.1 Commercial models for Phase 2 (researched 2026-07-29)

Probers stay local — a commercial prober would make the difficulty loop non-reproducible
(silent version upgrades), expensive (the loop iterates), and uncontrollable (no seed
guarantees). Commercial models appear **only** as Det image sources, and can never be
Table-A rows: their rates have no shelf life across silent updates, and the evaluated
pipeline calls no APIs.

| choice | model | why |
|---|---|---|
| ✅ | GPT Image 1.5 (OpenAI) | top of the human-preference Elo; strongest text rendering, so it also exercises family 5 |
| ✅ | Imagen 4 / Nano Banana 2 (Google) | elite tier, second lineage |
| ✅ | Seedream 4.5 / 5.0 Pro (ByteDance) | third lineage; alignment- and text-focused, incl. multilingual text |
| optional | Ideogram v3 | text-rendering specialist, targeted family-5 addition |
| ❌ | FLUX pro | same lineage as an eval generator; its style would leak between Gen and Det |
| ❌ | Midjourney V8.1 | no official API — dataset construction would not be reproducible |

Three lineages is the point: a detector cannot win Det by recognising one house style.

## 6. Open decisions

1. ~~Which commercial T2I models for Phase 2~~ — settled above; remaining: the API budget.
2. **Annotator plan** for the Det natural set and the Gen certification subset: how many
   annotators, and the target inter-annotator agreement. The pilot has *no* multi-annotator
   data, so nothing here is calibrated yet.
3. **Core pool composition** — which entities stay shared, which families get their own
   (DECISIONS.md #25 settled the principle, not the list).
4. **Two-step judging for the marking families** (DECISIONS.md #42--43): style is not
   fixed in the prompt, so families 2 and 3 need a style classifier before the family
   criterion. Report and certify $\kappa$ for both steps separately; a family whose style
   step is unreliable goes to human-only scoring, never to style-blind binary scoring.
5. **The κ threshold** below which a family is reported as human-only. Proposed 0.80,
   should be pre-registered before the runs.
6. **Venue**, which affects length and format: NeurIPS D&B fits a two-task benchmark best,
   ICLR is the broader alternative, and the T2I rescope makes *ACL a harder sell than it
   was when the paper had a text half.
