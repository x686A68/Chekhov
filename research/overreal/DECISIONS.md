# DECISIONS.md — OverReal / "When Chekhov's Gun Fires"

Design decisions, their rationale, rejected alternatives, and the literature map.
Kept because none of this survives in the paper source: `overrealization.tex` records
*what* to write, not *why it was chosen* or *what was ruled out*.

Companion files: `EXPERIMENT.md` (the core experimental design — what to execute),
`GOAL.md` (the pilot spec, now history), `pilot/REPORT.md` (the pilot's evidence),
`../../Chekhov_paper_ACL/overrealization.tex` (paper skeleton with per-paragraph `\todo`
plans).

---

## 1. Literature map — who occupies which family

The most expensive thing here to reconstruct. Verdict per family, with the work that
would collide.

| Family | Status | Occupying work |
|---|---|---|
| **1. Existence-canceling** | 🔴 heavily occupied | NEG-TTOI ("first negation T2I benchmark", 2k samples); [NEGATE](https://arxiv.org/pdf/2603.06533); [SpaceVLM](https://arxiv.org/pdf/2511.12331); [TNG-CLIP](https://arxiv.org/pdf/2505.18434); DCS-Bench; [Six-CD](https://arxiv.org/pdf/2406.14855) (safety-oriented concept removal). Text side: IFEval "forbidden words"; NegE / Modality / Tense errors in summarization taxonomies. Also [Ironic Negation in Transformers](https://arxiv.org/pdf/2511.12381) — the white-bear analogy is already used. |
| **2. Attribution** | 🟢 mostly open (generation side) | ToM benchmarks are dense (ToMi, FANToM, BigToM, Hi-ToM, OpenToM, XToM, TimeToM) but test *reasoning about* beliefs, not *suppression while generating*. Summarization lists attribution errors as a category. |
| **3. Figurative** | 🟠 partly occupied, wording collision | [IRFL](https://arxiv.org/pdf/2303.15445) (recognition, not generation); [T2I-ReasonBench](https://arxiv.org/pdf/2508.17472) has an "Idiom Interpretation" dimension and literally writes *"models often fail to suppress the literal rendering"* — our exact framing; SemEval-2025 Task 1 (ADMIRE); [I Spy a Metaphor](https://arxiv.org/abs/2305.14724); ViPE; GOME ("over-literalization"). |
| **4. Perspectival** | 🟢 open — the most novel cell | Repeated searches returned only camera control, multi-view consistency, storybook character consistency, and [PhyBench](https://arxiv.org/pdf/2406.11802) (physical commonsense). Nothing on whether depicted information is perceivable by an in-scene observer. |
| **5. Use–mention** | 🟢 open on the generation side | The recognition-side dual is well studied: typographic attacks — Goh et al. 2021 multimodal neurons, [Reading Isn't Believing](https://arxiv.org/pdf/2103.10480), [multi-image setting](https://arxiv.org/pdf/2502.08193) (NAACL 2025). Text-rendering work (TextDiffuser, [glyph-enhanced](https://arxiv.org/pdf/2403.16422)) treats legibility as the goal and never asks why the word is there. |
| **6. Relevance** | 🟢 ours, but with a close neighbour | [Contextual entrainment](https://arxiv.org/abs/2606.24077) (Liu & Chu 2026) is the closest prior work overall — it establishes the representation-level availability effect in closed QA. Also Shi et al. 2023 (GSM-IC), GSM-DC, [PI-LLM](https://arxiv.org/abs/2506.08184), Sinclair et al. 2022 structural priming. |

**Adjacent, opposite direction — must be distinguished explicitly in Related Work:**
POPE / CHAIR measure a VLM describing objects *absent from an image*; we measure
input-mentioned content wrongly realized in output.

**Evaluation precedents:** CLIPScore (known weak for compositionality),
[TIFA](https://arxiv.org/pdf/2303.11897), [VQAScore](https://arxiv.org/abs/2404.01291),
[T2I-CompBench(++)](https://arxiv.org/pdf/2307.06350),
[Commonsense-T2I](https://arxiv.org/abs/2406.07546) (closest methodology: paired
adversarial prompts + MLLM judge), [VBench-2.0](https://arxiv.org/abs/2503.21755).

**Theory drawn on:** Fauconnier, *Mental Spaces* (space builders) · Lakoff & Johnson
(conceptual metaphor) · Genette (focalization; diegetic/non-diegetic) · Grice (maxim of
relation), Sperber & Wilson · Kaup (two-step simulation of negation), Wegner (ironic
process) · Quine (use–mention), Saussure (signifier/signified), Peirce (icon/symbol) ·
Forceville (pictorial and multimodal metaphor — the precedent for applying
cognitive-linguistic frameworks to images) · Roediger & McDermott (DRM intrusion, the
source of our word "intrusion") · ANLI / Dynabench / AFLite (adversarial construction).

---

## 2. Settled decisions

**Scope and framing**
1. **Merge the two papers.** The text-only study covered one family in one modality and
   read thin; the taxonomy alone would have been descriptive with no mechanism. Merged,
   the mechanism from the first paper *explains* the taxonomy's cross-modal pattern.
2. **Title**: *When Chekhov's Gun Fires: A Cross-Modal Benchmark of Content That Should
   Not Be Realized*. "When" encodes the conditionality finding and promises no fix.
3. **Hook reframed around *dramatic* setup**: Chekhov's gun is a deliberate plant, while
   real user mentions are incidental; models treat the incidental as if it were planted.
4. **Position against contextual entrainment in one sentence**, not more: they establish
   availability, we ask about realization. Analogy: they measure "on the tip of the
   tongue", we measure when it is said aloud. Our own numbers show the dissociation
   (scoped QA: 7.0 nats availability but 1.4% surface intrusion; generative reading:
   comparable availability, 81%).

**Terminology**
5. **over-realization** = the construct (the only term we coin); **intrusion** = the
   measured event (continuous with the first paper, grounded in DRM); **base-space
   leakage** = the theoretical description.
6. **Benchmark name: OverReal** (previously Chekhov-Bench). The paper title carries the
   metaphor; the artifact carries the construct.

**Taxonomy**
7. **Six families**, ordered by *decreasing explicitness of marking*. Families 5 and 6
   were swapped from the first draft after noticing that relevance was never
   space-builder licensed either — the original "families 1–5 are space builders" claim
   was simply wrong.
8. **Three tiers of licensing**: space builders (1–4), use–mention (5), pragmatic
   relevance (6). Unity is claimed at the top level only — the input *contains*
   something it does not *license* realizing.
9. **Fauconnier as the single spine, with two extensions.** But calibrated: we do *not*
   claim to repair mental-spaces theory, which was never meant to cover relevance
   (Grice is canonical there). We claim only that space building does not exhaust the
   licensing mechanisms for this construct.
10. **Family 4 splits**: 4a occlusion (object behind a barrier, uses the core pool,
    carries the aligned cross-family comparison) and 4b diegetic legibility (inscribed
    information read by an in-scene observer — the showcase, but not pool-compatible).
11. **Unit of realization = information made accessible to the audience**, not merely an
    object being present. Without this, family 4 is not measurable.
12. **Two failure modes**: binary (1, 4–6) vs marking (2, 3).
13. **Section 3 is example-first.** Table 1 is "one entity, six failures" — unity is
    shown, not argued. Theory is compressed to one sentence attached to the example that
    motivates it. Table cells must be *real generations*, never invented.

**Benchmark design**
14. **S/P/A triples**, not pairs. `S = Sc + D(E)`, `P = Sc + E`, `A = Sc`. `P` prevents a
    model that realizes nothing from scoring perfectly; `A` measures the coincidental
    base rate *and* doubles as the compatibility filter for (entity, scenario) pairs.
15. **Licence sensitivity** `Δ = P(realize|P) − P(realize|S)` as the headline metric —
    the same paired effect size the mechanism analysis uses.
16. **Entity and scenario are crossed random factors** with Latin-square counterbalancing,
    not nested, so `(1|entity) + (1|scenario)` variance can be decomposed.
17. **Core pool (~12) spanning all six families and both modalities**, plus family
    extension pools (~18). The shared core is what makes cross-family and cross-modal
    comparison free of entity confounds.
18. **Family-specific entity:scenario ratios** — the load-bearing factor differs (4b
    varies carriers, not contents, since the letter's wording does not change whether it
    faces the camera). 60 items per cell → 720 combinations → 6,480 prompts.
19. **Openness ladder: three levels in both modalities, as a designed ordinal factor.**
    Not measured, therefore claims must be within-modality (slope inside each modality;
    most-constrained image vs most-open text).
20. **Evaluation protocol precedes the construction loop** in the text — the difficulty
    loop cannot select items without a score.
21. **Iterative difficulty construction** with difficulty labels; prober and evaluated
    model sets disjoint; pre- and post-filter subsets both released.
22. **T2I model: FLUX.1-dev** over SDXL, because family 5 needs legible text rendering.

**Repository**
23. Paper repo holds paper artifacts only; experiment material lives in the main repo
    under `research/`. Experiment scripts are versioned (they were previously ignored
    wholesale, though the paper's Reproducibility section names them).

---

## 3. Rejected alternatives

| Rejected | Why |
|---|---|
| **"unlicensed realization"** as the construct name | "Unlicensed" reads as *copyright* in a multimodal paper — a dangerous ambiguity. |
| **"over-generation" / "over-specification"** | Both are established NLG terms with different meanings (generate-and-rank / grammatical overacceptance; redundant attributes of a licensed referent). Must be explicitly distinguished in the paper. |
| **Three parallel theoretical traditions** as the organizing structure | Reads eclectic — "they grabbed whatever theory fit". Replaced by one spine plus two extensions. |
| **"The gun fired in the wrong direction"** discussion for family 4 | Unnecessary meta-commentary; replaced by a positive definition of the unit of realization. |
| **Embedding-based openness measurement** (mean pairwise distance) | Sentence-embedding space and CLIP/DINO space are different metric spaces with different scale and geometry, and CLIP has a documented modality gap — "text openness 0.09" and "image openness 0.09" would not denote the same thing. |
| **Making negation the headline** | Crowded (see §1). It stays as one family among six, with prior work cited. |
| **Contrastive decoding as the mitigation** | Tested and *failed*: contrasting against a task-free amateur left summarization intrusion at 0.96 → 0.95. The amateur does not specifically elevate the distractor. Connectivity filtering works instead (97.5% → 1.7% on HotpotQA). |
| **Attention-head causal localization** | Deferred — it is precisely the entrainment paper's strength, so doing it invites the overlap we are trying to avoid. |
| **Nine-paragraph dataset section** | Consolidated to six; two pieces relocated (E-type-varying → taxonomy; openness measurement → setup) and the release description folded into Reproducibility. |

---

## 3b. Decisions taken during the pilot (2026-07-26)

Numbering continues from §2. Evidence for each is in `pilot/REPORT.md`.

24. **Naturalness is a first-class constraint, traded against difficulty explicitly.**
    The pilot's family-3 items were built by *full* crossing of 4 entities x 3 scenarios,
    which is exactly what #16 warned against ("full crossing is impossible because many
    combinations are unnatural"). It produced items like *the auditor entering the office,
    as hungry as a wolf* — hard because they are odd, not because the phenomenon is hard.
    An item that no user would write is weak evidence whatever it shows. Three consequences:
    - **Return to Latin-square counterbalancing** (#16) rather than exhaustive crossing.
      The cost is that entity and scenario variance are no longer fully separable; the
      gain is that every item is defensible.
    - **Measure naturalness and report it beside difficulty**, so the trade-off is visible
      rather than buried in construction. Adversarial benchmarks rarely report it, so
      reporting it is both a small contribution and the cheapest defence against the
      obvious reviewer objection.
    - **Give the P5 difficulty loop a naturalness floor.** The loop mutates items until
      probers stop failing — every mutation step pushes toward the unnatural. Without a
      floor it converges on prompts models cannot parse rather than prompts models should
      handle and do not. Candidates below the floor are discarded, not kept as hard items.

    Pilot items are kept as-is for now; higher-quality natural items are to be added
    alongside them rather than replacing them.

25. **The shared core pool is not required across all families.** #17 justified it as
    protection against entity confounds, but the pilot found essentially no entity
    variance: all four candidates failed exactly the same two of six families in text.
    Paying for a confound that is not there costs the families their expressiveness.
    Keep a *small* core pool to serve Table 1 and the 4a aligned comparison; let the other
    families use pools chosen for what they actually test. #10 already conceded this for 4b.

26. **Family 5 needs text-bearing carriers and a generator that writes real words.**
    Two constructions returned 0.00. The crate ("a crate of elephants") gave the model no
    occasion to render text at all. Text-bearing carriers (book, letter, newspaper) do
    force text — a cover shows a title, not the word BOOK — but FLUX.1-dev answers with
    pseudo-text: blank sheets, unreadable page texture, garbled mastheads. It makes no
    lexical commitment, so the use-mention choice never arises. The blocker is the
    generator, not the family. Retry on a strong text-rendering model (Qwen-Image,
    Apache-2.0, local) before deciding the cell is dead.

27. **Family 6's S condition must be genuinely incidental.** The pilot marked the
    irrelevant mention with the words "Unrelated aside:", which turns a family defined by
    the *absence* of marking into a marked one. The correct form is a real stretch of
    unrelated conversation followed by a "by the way" request. In text this is a
    multi-turn setup; T2I has no conversation, so the image side needs either a
    conversational image generator or an honest note that the modality cannot host the
    family as specified.

28. **Explicitness of marking becomes a within-family factor (implicit / explicit).**
    The taxonomy's own ordering is by "how explicitly the input marks the suppression",
    which is currently stipulated between families. Building an implicit/explicit contrast
    inside each family makes that ordering measurable, and gives family 6 the
    implicit/explicit noise split it needs. First evidence (4a v2, n=12): explicit marking
    failed *more* than implicit (0.83 vs 0.67), the ironic-process direction — a hypothesis
    to design for, far too small to claim.

29. **Family 4a's construction is replaced.** v1 put the camera nowhere in particular, made
    a correctly suppressed S image identical to A, and could not distinguish "ignored the
    occlusion" from "never drew the barrier". v2 puts the camera at the observer, so the
    audience's access and the observer's access coincide (#11), and adds a barrier-validity
    question. v1 saturated at 1.00 with a negative delta; v2 leaves headroom and grades the
    failure.

30. **The VLM judge belongs to the construction loop, not to final evaluation.** Its
    reliability failures (families 3 and 4b) are therefore less costly than they look, but
    not free: a judge that *understates* over-realization reads failing items as passing,
    so the difficulty loop selects for items that confuse the judge rather than items that
    defeat the generator. The bias lands on item selection, where it is harder to detect
    later than a wrong headline number would be.

## 3c. Decisions from the overnight extension (2026-07-27)

31. **The VLM judge's failures were a question-design problem, not a model limit.** One
    plain positive binary about depicted content scores families 2, 3, 4b, 4a2 and 5b at
    kappa 0.94–1.00, against 0.23–0.80 for the multiple-choice form, with the last two
    families out of sample. Four wording faults are documented in REPORT.md §4.3: a third
    hedged option, listing embedding devices (every generated image *is* a picture), the
    words "real"/"real live" (which make the judge answer about ontological status), and
    negative framing (agreed with 89% of the time). The rule for every family-specific
    judge: **one positive question about depicted content, plainest words, no options.**
32. **Explicitness of marking has a real but moderate effect — the first estimate was
    inflated by a validity flaw.** ~~In text, family 4a v2 goes from Δ ≈ 1.00 with an
    explicit perceptual marker to Δ ≈ 0.03 with only a spatial relation, for all three
    models; the strongest single factor found so far.~~ **Corrected 2026-07-28:** v2's
    implicit condition said the entity was "on the far side" of "a tall wooden fence",
    which does not entail invisibility — an elephant is taller than a fence, so drawing it
    above the fence is a faithful reading, not a failure. S_imp was therefore a weaker
    *fact*, not a weaker *marking* of the same fact, and much of the gap was legitimate
    behaviour scored as failure. v3 gives every barrier an absolute height that exceeds
    every entity and places the entity "entirely" beyond it, so occlusion follows from the
    geometry alone. The S_imp − S_exp gap in text falls from 0.92 to **0.40** on average
    and stops being consistent across models (0.42 / 0.08 / 0.71 for llama-3.1-8b /
    qwen3-8b / qwen3-32b). Still worth building the ladder into every family, but as one
    factor among several, not as the headline. n = 24 items x 3 models.
33. **Oblique realization is a third outcome and needs its own category.** The entity
    appears as a depiction inside the scene — a child's drawing of a tiger, a calendar, a
    mural, an anthropomorphic blend — on a carrier the scene independently licenses. It
    occurs under P as well as under S (0.08–0.43), so it is not a suppression strategy but
    how the model reconciles a mentioned entity with a scene it does not fit. Under
    DECISIONS #11's definition it is a failure. It breaks the binary/marking dichotomy
    (#12) and it is what the VLM judge kept reaching for when it answered "only a picture".
34. **A text-to-image model has no context/request distinction, so format cues become
    scene content.** Serialising family 6's conversation as a `User:`/`Assistant:`
    transcript made FLUX draw comic strips — in the A condition too. The same content in
    prose works. Relevance can be tested in the image modality, but only if nothing in the
    prompt signals that the input is a transcript.
35b. **Oblique realization is a human-annotation category only.** It stays in the
    annotation scheme and in the released labels, but it is not scored automatically:
    telling "a child's drawing of a tiger" from "a tiger" reliably enough for a machine
    judge is harder than the rest of the protocol put together, and a bad detector would
    contaminate every family's headline rate. The regex flag in `score_text.py` is a
    reporting aid for reading generations, not a scorer. Revisit only if a human-annotated
    subset shows the category is frequent enough to change conclusions.

35. **Per-token perplexity is not a naturalness measure.** It is dominated by the
    predictability of the scenario phrase and is nearly blind to whether the licensing
    device fits, ranking the pilot's most awkward figurative item among its most natural.
    Naturalness stays a human-rated field on a sample until a conditional fit measure is
    validated; it must not be wired into the P5 loop as a floor before then. Superseding
    the automatic-filter half of #24.

## 3d. Scope decision: T2I-only, generation plus detection (2026-07-29)

36. **The benchmark is text-to-image only.** The pilot settles this on evidence: in text,
    four of the six families sit at exactly 0.00 — existence-canceling, attribution,
    figurative and use-mention — and family 6 falls to 0.01 once its items are natural.
    Only 4a (0.21) and 4b (0.50) produce failures. A cell at the floor cannot support a
    difficulty loop, cannot rank models and cannot show progress, so two thirds of a
    cross-modal benchmark would be dead weight. The measurement instruments are asymmetric
    too: the image-side judge protocol reaches kappa 0.94–1.00 with out-of-sample
    confirmation (#31), while the text-side scope scorer produced wrong numbers three
    times in one session and has never been validated. Dropping text removes the only
    unvalidated instrument.

37. **Text survives as a control, not as half the benchmark.** ~36 items per family, run
    only to establish the modality asymmetry — identical stimuli, 0.00 in text against
    0.58 in images for family 3 — reported in one section with one table. No openness
    ladder, no difficulty loop, no released text subset. Measuring an asymmetry costs a
    fraction of what building a modality costs.

38. **The mechanism half is dropped.** This reverses the merge rationale in #1, which
    argued that the taxonomy alone "would have been descriptive with no mechanism" and
    that the first paper's mechanism explained the cross-modal pattern. All of that
    analysis — the entrainment comparison, the 7.0 nats availability against 1.4% surface
    intrusion, connectivity filtering at 97.5% → 1.7% — is text-side and goes with it.
    The replacement for it is #39: the paper stops being a taxonomy plus a mechanism and
    becomes a taxonomy plus a second task.

39. **The benchmark has two halves: generation and detection.** Generation is the current
    S/P/A design. Detection presents a suppression prompt together with a candidate image
    and asks whether the image over-realizes. The pilot is the argument for including it:
    the same VLM moved from kappa 0.23 to 0.94 on the same images purely from how the
    question was worded (#31), so whether a model can recognise over-realization is an
    open problem in its own right, not a solved preliminary.

40. **Detection items are constructed so that most labels do not depend on judging the
    generator.** Circularity enters in three places and is handled separately:
    - *Labels from a VLM judge* would test detectors against a detector. *Labels are
      human only.* The pilot is the precedent: hand inspection exposed the judge, not the
      reverse.
    - *Condition used as label* is simply wrong, not merely circular: an S image may be a
      correct suppression and a P image may have omitted the entity.
    - *Single-generator items* let a detector win by recognising one model's style.

    The construction that removes the first two: **pair prompts with images across
    conditions**, exploiting the shared seed that already makes S, P and A near-identical
    in composition.

    | item | label | where the label comes from |
    |---|---|---|
    | S prompt + **P image** | over-realizes | construction — the image was generated from an explicit request for E |
    | S prompt + **A image** | does not | construction — measured base rate is 0.00 |
    | S prompt + **S image** | either | **human annotation required** |

    The first two classes are large, free and carry certain labels without anyone judging
    whether the generator erred. The third is the naturally-occurring distribution and is
    the only part needing annotation. Reporting them separately gives a
    construction-guaranteed core plus a natural-distribution set. The A-condition base
    rate of 0.00 measured in the pilot is what licenses A images as certain negatives.

41. **Two controls are load-bearing for the detection half; without them it does not
    stand.** Run the detector (a) with the prompt withheld and (b) with a mismatched
    prompt. If either scores above chance, the items leak — the detector is reading
    generator style or prompt length artefacts rather than the licensing relation. Also
    required: at least two generators, and the evaluated detector must not be the model
    whose labels were used anywhere in construction.

---

## 4. Open risks

1. **Judge reliability for families 2 and 4b** — the two judgements that are not simple
   presence checks. Pilot Q2. If the judge cannot do 4b geometry, the paper's headline
   example is at risk.
2. **Family 3 may be at ceiling in text** — pilot Q3. If so it becomes a modality
   asymmetry finding rather than a full family.
3. **Table 1's entity** — no entity is yet known to fail in all six families; elephant is
   expected to be weak in 4a and 5. Pilot Q1.
4. **Cross-modal openness confound** — image generation is intrinsically freer than
   question answering. Handled by keeping claims within-modality; revisit if a stronger
   claim is wanted.
5. **Adversarial filtering bias** — items overfit to the probers. Mitigated by disjoint
   prober/eval sets and by releasing both subsets.
6. **Self-overlap with the first paper** if it is ever submitted separately. Current plan
   is full absorption, which removes the problem.
7. **Wording collision with T2I-ReasonBench**, which already uses "fail to suppress the
   literal rendering" for idioms. Family 3 must cite it prominently and claim only the
   unification.
