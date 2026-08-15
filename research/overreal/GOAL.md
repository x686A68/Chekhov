# GOAL.md — OverReal pilot

Task specification for the pilot study that preceded building the OverReal benchmark.
**Status: the pilot has run** (results in `pilot/REPORT.md`; answers to Q1–Q3 noted
inline below). The paper — `Chekhov_paper_ICLR/overrealization.tex`, the submodule — is
now the authority on the construct and the taxonomy, and `EXPERIMENT.md` on the
benchmark design; this file is kept as the pilot's spec, with its facts corrected to
match the paper. Section 3 of the paper defines the construct and Table 1; Section 4
defines the benchmark.

---

## 1. Background in one page

We study **over-realization**: content that appears in the input but should *not* be
realized in the output. A mention is not a licence to realize. The name comes from
Chekhov's principle (a gun on the wall in act one must fire in act three) — except that
real user inputs are not dramatic setups, and models discharge them anyway.

Six **families**, ordered by how explicitly the input marks the suppression (numbering
per the paper §3.2; the pilot ran under an older scheme — see the note below the table):

| # | Family | Licensing device | Example (image) | Failure |
|---|---|---|---|---|
| 1 | Existence-canceling | *no, without, never* | "a hotel lobby with **no** elephant in it" | elephant present |
| 2 | Attribution | *believes, thinks, remembers, dreams* | "Maya at the ridge, **believing** there is an elephant at the ridge" | a real elephant, unmarked |
| 3 | Figurative | *like, as … as, resembles, -like* | "the porter, **as heavy as** an elephant" | a literal elephant beside him |
| 4 | Perspectival | *behind, on the far side, cannot see*; viewpoint frame | "an elephant **entirely on the far side** of a four-metre wall"; "Anna reads a letter saying 'I'm sorry'" | elephant shown anyway; letter faces the camera, not Anna |
| 5 | Relevance | none (pragmatic) | [earlier talk of an elephant] "**Anyway** — a school gymnasium after the game" | elephant on the court |
| 6 | Other | — | residual class for over-realization that fits none of 1–5 | — |

Families 1–4 are licensed by *space builders* (Fauconnier, Mental Spaces); family 5 lies
outside mental-spaces theory and is licensed by Gricean relevance; family 6 is a
residual class, not a construction. Family 4 contains two kinds of item — **occlusion**
(an entity behind a barrier; the pilot's 4a) and **legibility** (inscribed content that
the in-scene observer, not the camera, has access to; the pilot's 4b). Families 1 and 5
are **binary** (should it appear at all); families 2 and 3 are **marking** failures (it
may appear, but only as explicitly embedded — a thought bubble, or a stylized shape);
family 4 is **geometric** (may appear only as the viewpoint licenses).

**Changes from the pilot's numbering.** The pilot used 4a/4b as separate cells, 5 =
use–mention, 6 = relevance. Use–mention never reproduced on the piloted generator
(0.00 for every entity in both modalities — FLUX writes pseudo-text and makes no
lexical commitment; REPORT §1, DECISIONS.md #26) and left the taxonomy; relevance
renumbered 6 → 5, and "other" was added as 6. Pilot artifacts (`pilot/`, REPORT.md,
DECISIONS.md) still use the old numbering.

**Item structure.** Every item is a scenario `Sc`, an entity `E`, and the family's
licensing device `D`, assembled three ways:

```
S (suppression) = Sc + D(E)   "a room with no elephant"      → E should NOT be realized
P (positive)    = Sc + E      "a room with an elephant"      → E SHOULD be realized
A (absent)      = Sc          "a room"                       → measures coincidental base rate
```

`P` is essential: without it, a model that realizes nothing scores perfectly.
(Since the pilot, the design issues `S` at two cue strengths — explicit `S_exp`, the
suppression stated, and implicit `S_imp`, left to inference — making explicitness a
measured within-family factor; see EXPERIMENT.md §1.1.)
Headline metric — **licence sensitivity**:

```
Δ = P(realize | P) − P(realize | S)      range [−1, 1]
```

`Δ = 1` is perfect discrimination; `Δ = 0` means the licensing device was ignored.

---

## 2. What this pilot must answer

Three blocking questions. Everything else is secondary.

**Q1 — Which entity fails in all families?**
Table 1 of the paper is "one entity, N failures", and its cells must be **verbatim real
generations**, not invented placeholders. Test 3–4 candidate entities across all
families and report which one (if any) fails everywhere. If none does, say so — the
table may need two entities.

> *Answered (REPORT §1): no entity failed all six of the pilot's families — elephant
> and tiger each failed 5 of 6, blocked only by use–mention, which was 0.00 for every
> entity in both modalities. Use–mention subsequently left the taxonomy, and the
> paper's Table 1 is now "one entity, five failures", filled with verbatim FLUX.1-dev
> generations.*

**Q2 — Can a VLM judge score family 2 and family 4's legibility items reliably?**
These are the two judgements that are not a simple presence check.
- Family 2 asks: is the snake rendered *as an actual object in the scene*, or as
  *explicitly embedded* (thought bubble, dream haze, framed inset)?
- The legibility items ask: could the person in the image actually perceive the
  depicted information — i.e. is the letter angled toward the reader or flat to the
  camera?

Generate images, have the VLM judge answer the structured question, then **inspect the
images directly** and record where the judge disagrees with what is visibly the case.
If the judge cannot do the legibility geometry, those items need a different protocol
(or human-only scoring), and the paper's headline example is at risk.

> *Answered (REPORT §4): reliability is a property of the question wording, not of the
> model — the same 7B judge moved from κ = 0.23–0.80 (multiple choice with a hedged
> option) to κ = 0.94–1.00 (single positive binary) on the same images. This became
> the paper's protocol finding and the motivation for the detection task.*

**Q3 — Is family 3 at ceiling in text?**
LLMs handle idioms well, so the figurative family may show no measurable effect in text.
If so, that is a legitimate finding (a modality asymmetry), but it changes whether
family 3 gets a full treatment or a footnote.

> *Answered (REPORT §2; DECISIONS.md #36–37): yes — and not only family 3. Four of the
> pilot's six families sat at exactly 0.00 in text (family 3: 0.00 in text against
> 0.58 in images). The benchmark became text-to-image only; text survives as a
> ~36-item-per-family control that establishes the modality asymmetry.*

Secondary, but record them: per-item wall-clock and VRAM cost (to correct the projected
6,480-prompt budget); and whether FLUX renders text legibly enough for the items that
must bear text. *(Answer: no — FLUX writes pseudo-text; Qwen-Image is the retry
candidate. DECISIONS.md #26.)*

---

## 3. Environment

**Hardware.** One or two 80GB GPUs is enough. *On the original machine only GPUs 6 and 7
were permitted — confirm the constraint on the new machine before launching.*

**Models** (all open-weight, downloaded from HuggingFace):

| Role | Model | Notes |
|---|---|---|
| Text generation | `Qwen/Qwen3-8B`, `Qwen/Qwen3-32B`, `meta-llama/Llama-3.1-8B-Instruct` | disable thinking mode; greedy decoding |
| Text-to-image | `black-forest-labs/FLUX.1-dev` | chosen over SDXL for legible text rendering, which the text-bearing items need (pilot verdict: still pseudo-text; DECISIONS.md #26) |
| VLM judge | `Qwen/Qwen2.5-VL-7B-Instruct` | structured yes/no and multiple-choice questions |
| Embeddings (optional) | `BAAI/bge-large-en-v1.5` | semantic-intrusion checks |

**Python**: `torch`, `transformers`, `diffusers`, `accelerate`, `matplotlib`, `datasets`.

**Hard constraint: no external model APIs.** Everything runs on local weights. This has
been a standing constraint of the whole project and the paper claims it.

---

## 4. Tasks

### Phase 0 — text pilot (no downloads needed if the text models are cached)

For each family (the pilot's six, including the since-dropped use–mention), hand-write
**10 items** with all three conditions, using 2–3 candidate core entities. Run all three text models, greedy decoding.

Scoring (current numbering): entity string match (word-form normalized) for families 1,
3 and 5, and for family 4's occlusion items; presence of the attribution phrase for
family 2; for family 4 in text, whether the narration reports something outside the
focalizing character's access. (The pilot additionally scored the now-dropped
use–mention family by literal template slot / instruction echo.)

Report per family: realization rate under S, under P, under A, and `Δ`.

### Phase 1 — download the image stack

FLUX.1-dev and Qwen2.5-VL. Run in the background while Phase 0 proceeds.

### Phase 2 — image pilot

For each family, **10 items × (S, P)** = 120 images. Same entities as Phase 0 so the
two modalities are comparable. Save every image with its prompt and condition.

### Phase 3 — judge reliability

Run the VLM judge on all Phase 2 images. For family 2 and family 4's legibility items,
additionally inspect every image directly and record agreement with the judge. Report agreement per family.

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
  table1_candidates.md               verbatim generations for the Table 1 cells
                                     (five in the current taxonomy), per candidate entity
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
