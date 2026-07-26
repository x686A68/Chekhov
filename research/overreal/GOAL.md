# GOAL.md — OverReal pilot

Task specification for the pilot study that precedes building the OverReal benchmark.
Written to be run on a fresh machine. **Read `Chekhov_paper_ACL/overrealization.tex` first** (the paper submodule) — it is the
paper skeleton, and its `\todo[inline]` blocks contain the full design rationale.
Section 3 defines the construct and Table 1; Section 4 defines the benchmark.

---

## 1. Background in one page

We study **over-realization**: content that appears in the input but should *not* be
realized in the output. A mention is not a licence to realize. The name comes from
Chekhov's principle (a gun on the wall in act one must fire in act three) — except that
real user inputs are not dramatic setups, and models discharge them anyway.

Six **families**, ordered by how explicitly the input marks the suppression:

| # | Family | Licensing device | Example (image) | Failure |
|---|---|---|---|---|
| 1 | Existence-canceling | *no, without, if, might* | "a room with **no** elephant" | elephant present |
| 2 | Attribution | *believes, claims, dreamed* | "Maya **thought** the boulder was an elephant" | a real elephant, unmarked |
| 3 | Figurative | *like, as,* idiom | "a man **as heavy as an elephant**" | an elephant beside him |
| 4a | Perspectival (occlusion) | *behind, offscreen* | "an elephant **behind the high wall**" | elephant fully visible |
| 4b | Perspectival (legibility) | viewpoint frame | "Anna reads a letter saying 'I'm sorry'" | letter faces the camera, not Anna |
| 5 | Use–mention | semiotic level | "a crate of **elephants**" | crate stamped with the word ELEPHANT |
| 6 | Relevance | none (pragmatic) | [earlier mention of an elephant] "draw something relaxing" | elephant appears |

Families 1–4 are licensed by *space builders* (Fauconnier, Mental Spaces); 5 by the
use–mention distinction; 6 by Gricean relevance. Families 1 and 4–6 are **binary**
(should it appear at all); families 2 and 3 are **marking** failures (it may appear, but
only as explicitly embedded — a thought bubble, or a hedge like "she believed").

**Item structure.** Every item is a scenario `Sc`, an entity `E`, and the family's
licensing device `D`, assembled three ways:

```
S (suppression) = Sc + D(E)   "a room with no elephant"      → E should NOT be realized
P (positive)    = Sc + E      "a room with an elephant"      → E SHOULD be realized
A (absent)      = Sc          "a room"                       → measures coincidental base rate
```

`P` is essential: without it, a model that realizes nothing scores perfectly.
Headline metric — **licence sensitivity**:

```
Δ = P(realize | P) − P(realize | S)      range [−1, 1]
```

`Δ = 1` is perfect discrimination; `Δ = 0` means the licensing device was ignored.

---

## 2. What this pilot must answer

Three blocking questions. Everything else is secondary.

**Q1 — Which entity fails in all six families?**
Table 1 of the paper is "one entity, six failures", and its cells are currently invented
placeholders. They must be replaced with **verbatim real generations**. Elephant is the
current candidate but is expected to be weak in 4a and 5. Test 3–4 candidate entities
across all six families and report which one (if any) fails in all six. If none does,
say so — the table may need two entities.

**Q2 — Can a VLM judge score families 2 and 4b reliably?**
These are the two families whose judgement is not a simple presence check.
- Family 2 asks: is the snake rendered *as an actual object in the scene*, or as
  *explicitly embedded* (thought bubble, dream haze, framed inset)?
- Family 4b asks: could the person in the image actually perceive the depicted
  information — i.e. is the letter angled toward the reader or flat to the camera?

Generate images, have the VLM judge answer the structured question, then **inspect the
images directly** and record where the judge disagrees with what is visibly the case.
If the judge cannot do 4b geometry, that family needs a different protocol (or
human-only scoring), and the paper's headline example is at risk.

**Q3 — Is family 3 at ceiling in text?**
LLMs handle idioms well, so the figurative family may show no measurable effect in text.
If so, that is a legitimate finding (a modality asymmetry), but it changes whether
family 3 gets a full treatment or a footnote.

Secondary, but record them: per-item wall-clock and VRAM cost (to correct the projected
6,480-prompt budget); and whether FLUX renders text legibly enough for OCR to score
family 5.

---

## 3. Environment

**Hardware.** One or two 80GB GPUs is enough. *On the original machine only GPUs 6 and 7
were permitted — confirm the constraint on the new machine before launching.*

**Models** (all open-weight, downloaded from HuggingFace):

| Role | Model | Notes |
|---|---|---|
| Text generation | `Qwen/Qwen3-8B`, `Qwen/Qwen3-32B`, `meta-llama/Llama-3.1-8B-Instruct` | disable thinking mode; greedy decoding |
| Text-to-image | `black-forest-labs/FLUX.1-dev` | chosen over SDXL for legible text rendering, which family 5 needs |
| VLM judge | `Qwen/Qwen2.5-VL-7B-Instruct` | structured yes/no and multiple-choice questions |
| Embeddings (optional) | `BAAI/bge-large-en-v1.5` | semantic-intrusion checks |

**Python**: `torch`, `transformers`, `diffusers`, `accelerate`, `matplotlib`, `datasets`.

**Hard constraint: no external model APIs.** Everything runs on local weights. This has
been a standing constraint of the whole project and the paper claims it.

---

## 4. Tasks

### Phase 0 — text pilot (no downloads needed if the text models are cached)

For each of the six families, hand-write **10 items** with all three conditions, using
2–3 candidate core entities. Run all three text models, greedy decoding.

Scoring: entity string match (word-form normalized) for families 1, 3, 4a, 6; presence
of the attribution phrase for family 2; literal template slot / instruction echo for
family 5; for family 4 text, whether the narration reports something outside the
focalizing character's access.

Report per family: realization rate under S, under P, under A, and `Δ`.

### Phase 1 — download the image stack

FLUX.1-dev and Qwen2.5-VL. Run in the background while Phase 0 proceeds.

### Phase 2 — image pilot

For each family, **10 items × (S, P)** = 120 images. Same entities as Phase 0 so the
two modalities are comparable. Save every image with its prompt and condition.

### Phase 3 — judge reliability

Run the VLM judge on all Phase 2 images. For families 2 and 4b, additionally inspect
every image directly and record agreement with the judge. Report agreement per family.

*This is a first-pass sanity check, not a substitute for human annotation — the real
benchmark still needs annotated subsets with reported inter-annotator agreement.*

---

## 5. Deliverables

Write everything to disk as it is produced; do not hold results in context.

```
pilot/
  text/<family>/results.jsonl        one line per item: id, family, entity, scenario,
                                     condition, prompt, output, realized (bool)
  text/<family>/summary.json         rates under S/P/A, Δ, n
  images/<family>/<item>_<cond>.png  every generated image
  images/<family>/results.jsonl      judge verdicts + direct-inspection verdicts
  judge_agreement.json               per-family agreement between judge and inspection
  table1_candidates.md               verbatim generations for the six Table 1 cells,
                                     per candidate entity
  REPORT.md                          answers to Q1/Q2/Q3, cost measurements,
                                     and a go/no-go recommendation per family
```

---

## 6. Rules

1. **Minimal difference is the whole design.** `S`, `P` and `A` must be identical except
   for the licensing device. Match length and framing; do not let the S prompt be longer
   or more emphatic.
2. **No keyword leakage.** The target entity string must not appear in the `P` or `A`
   context, nor in any instruction. Check this programmatically.
3. **Check `A` first.** If the entity appears spontaneously in the `A` condition, the
   (entity, scenario) pair is unusable — drop it and record why. `A` is both a validity
   check and a filter.
4. **Report negative results plainly.** A family that shows no effect, or a judge that
   cannot score a family, is a finding and determines the paper's scope. Do not tune the
   prompts until an effect appears; if prompts are revised, report both versions.
5. **Suspect surprisingly clean results.** If a rate is 0% or 100%, inspect the raw
   outputs before believing it. Two earlier bugs in this project were found this way.
6. Redirect long generation logs to files; keep only summaries in the working context.
