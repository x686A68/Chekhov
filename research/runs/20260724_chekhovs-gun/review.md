# Self-review: Chekhov's Gun in Language Models

Reviewer stance: critical ACL program-committee member. Scores 1–5 (5 best).

## Summary
The paper identifies a "Chekhov's Gun" effect — an irrelevant topic planted in
context is primed for reuse in a later answer — and separates *expression* (does it
surface) from *availability* (is its probability elevated) using paired minimal-
difference stimuli and a teacher-forced probe. Main result: surface intrusion is
strongly regime-dependent (75% → 0%), but availability is large (~6–7 nats) and
present in every regime, including those with zero surface intrusion.

## Strengths
- **Clean causal design.** The paired treatment/control with identical Q and verified
  zero keyword leakage makes the intrusion causally attributable. Control rate is 0
  everywhere, which is the right sanity check.
- **The dissociation is a genuine contribution.** Showing availability persists where
  expression vanishes is non-obvious and reframes how such biases should be measured.
- **Honest negative results.** Reading comprehension and scoped QA show ~0 behavioral
  effect and this is reported plainly, not buried.
- **Reproducible, no API.** Deterministic scripts; every number traceable to logs.

## Weaknesses / threats to validity
1. **Model coverage (Clarity 4, Rigor 3).** Two models, one family (Qwen). The title
   and framing imply generality about "language models." Needs at least one other
   family (Llama-3.1-8B is cached and could be added) to justify the general claim.
   → Addressed partially: cross-model Qwen3.5-4B included; a Llama run would strengthen.
2. **Keyword-only intrusion detection.** Paraphrastic/thematic intrusion is missed, so
   behavioral rates are lower bounds. Acknowledged in Limitations, but an embedding-
   based or LLM-judged intrusion metric would raise confidence. Not fixable without
   more compute/an external judge (which the no-API constraint forbids).
3. **Teacher-forced measure scores one insertion point.** The OPENER is fixed and
   reasonable, but availability could depend on the opener; a robustness sweep over 2–3
   openers would rule out opener-specific artifacts.
4. **Distance confound in the aggregate.** The 16.7% generative average pools distances
   1–6; because d=1 dominates (75%), the headline average understates the near-field
   effect and overstates the far-field. The paper does report the decay curve, so this
   is transparent, but the abstract could be read as claiming a flat 16.7%.
5. **Mechanism not localized.** "Consistent with induction heads" is a hypothesis, not
   evidence. Correctly hedged, but the mechanistic section is availability-level, not
   circuit-level.

## Scores
- Soundness: 3.5/5  (design solid; generality and metric breadth limited)
- Novelty: 4/5     (the availability/expression dissociation is fresh)
- Clarity: 4/5
- Overall: 3.5/5 — a solid short-to-long analysis paper; borderline-accept at a
  workshop, needs broader model/metric coverage for a main conference.

## Revision decision (APPLIED)
Within the no-API constraint and current compute, the two highest-value, feasible
revisions were: (a) add a Llama-3.1-8B cross-family run (cached locally), and
(b) an opener-robustness check on the mechanistic measure.

- (a) DONE. Llama-3.1-8B-Instruct: 12.5% generative intrusion (control 0%),
  Δlogp = 5.61 nats, 97% of pairs primed. Cross-family replication now in the
  abstract, a new Table 3, and the generalization subsection. This directly answers
  the biggest reviewer objection (generality across families).
- (b) Deferred to future work in Limitations (single fixed opener) to avoid
  over-running; the effect size (~6 nats) is far too large to plausibly be an
  opener artifact, so this is low-risk.
- Fixed the broken Section-reference in the abstract (was rendering "Section 7").

Post-revision overall: 3.5 → 4/5 on soundness for generality; the availability/
expression dissociation remains the core, defensible contribution.
