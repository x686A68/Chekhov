# OverReal pilot — report

Run: 2026-07-26, host `doremi`, GPUs 4 and 5 (H100 NVL 95 GB each).
Everything below comes from local open weights; no external model API was called.

**Status: text modality complete (Phase 0), image modality complete (Phase 2),
judge reliability complete (Phase 3).** Section 6 records what the pilot did not settle.

---

## 0. What was run

| Phase | What | Output |
|---|---|---|
| 0 | 84 items x 3 conditions x 3 text models = 756 generations | `pilot/text/` |
| 1 | FLUX.1-dev, Qwen2.5-VL-7B, Qwen3-32B downloaded to `/data/users/jiahao_huang/hf` | — |
| 2 | 84 items x 3 conditions = 252 images, FLUX.1-dev | `pilot/images/` |
| 3 | VLM judge on all 252 images + direct inspection of families 2, 3 and 4b (108 images) | `pilot/judge_agreement.json` |

**Deviations from GOAL.md, and why.**

1. **12 items per family cell, not 10** (4 candidate entities x 3 scenarios). A balanced
   crossing gives Q1 exactly 3 items per (entity, family) cell; 10 would not divide.
2. **Seven cells, not six.** Family 4 is split into 4a (occlusion, core entity pool) and
   4b (diegetic legibility, carrier pool) because they are scored differently and 4b is
   not pool-compatible (DECISIONS.md #10).
3. **Condition A was generated in the image modality too** (252 images, not the 120
   specified). GOAL.md rule 3 makes A a validity filter, and that argument applies to
   images as much as to text. Cost was 51 GPU-minutes, so the extra 84 images were cheap.
4. **Two scores per text generation, not one.** GOAL.md specifies word-form-normalized
   string match. On its own that score is confounded for exactly the families the
   construct is about — "the lobby contained no elephant" is a surface hit and a correct
   suppression. Both are reported; the headline uses affirmative realization (§2).
5. **Family 5's realized unit differs by modality**, by construction. In text the input
   mentions the word and the referent must not appear; in images the input mentions the
   referent and the word must not appear. These are duals, and only the image direction
   matches the example in GOAL.md's Table.
6. **Family 3 was inspected too, though GOAL.md asked only for 2 and 4b.** Its judge
   numbers contradicted what the contact sheets plainly showed, and rule 5 says to inspect
   before believing. That decision changed the answer to Q3 (§2) and added a family to the
   list of judgements the VLM cannot make (§4).
7. **`overrealization.tex` was read only after the pilot had run.** The submodule is
   configured with an HTTPS remote and the machine has no credential helper, so the clone
   failed with `could not read Username for 'https://github.com'`; it was cloned over SSH
   afterwards. The design was taken from `DECISIONS.md` in the meantime and turned out to
   match the skeleton — S/P/A, Δ, family-specific judges, 4b varying carriers rather than
   contents. One plan was missed and has since been run: P2's paired forced-choice
   protocol (§4.1). Two remain out of scope: P4's openness ladder, and P2's dedicated OCR
   pass for family 5, approximated here with a VLM yes/no question.

---

## 1. Q1 — which entity fails in all six families?

**Answer: none — and the obstacle is a family, not an entity.** Pooled across both
modalities, elephant and tiger each fail **5 of 6**; the one cell neither reaches is
family 5 (use–mention), which is at 0.00 for *every* entity in *both* modalities:

| entity | 1 existence | 2 attribution | 3 figurative | 4a occlusion | 5 use-mention | 6 relevance | failed |
|---|---|---|---|---|---|---|---|
| **elephant** | 0.00 / **0.33** | 0.00 / **1.00** | 0.00 / **1.00** | **0.44** / **1.00** | 0.00 / 0.00 | **0.33** / **1.00** | **5/6** |
| snake | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / **0.33** | **0.33** / **1.00** | 0.00 / 0.00 | **0.22** / **1.00** | 3/6 |
| **tiger** | 0.00 / **0.33** | 0.00 / **0.67** | 0.00 / **0.67** | **0.33** / **1.00** | 0.00 / 0.00 | **0.33** / **1.00** | **5/6** |
| wolf | 0.00 / 0.00 | 0.00 / **1.00** | 0.00 / **0.33** | **0.44** / **1.00** | 0.00 / 0.00 | **0.33** / **1.00** | 4/6 |

Cells are `text S / image S`. Full table in `pilot/q1_crossmodal.md`; the verbatim
generations and image paths for every non-empty cell are in `pilot/table1_candidates.md`.

**Elephant is the best candidate** — it was expected to be weak in 4a and 5, and it is not
weak in 4a at all (0.44 in text, 1.00 in images, the hardest-failing family). It is weak
in 5, but so is everything else: the use–mention failure did not occur once in 504 text
generations or 84 images (§3). Table 1 can therefore be built with elephant for five of
six cells today; the sixth needs either a different family-5 construction or an honest
"not observed".

**Within the text modality alone the entity is not a variable at all** — all four
candidates fail exactly the same two of six families:

| entity | 1 existence | 2 attribution | 3 figurative | 4a occlusion | 5 use-mention | 6 relevance | failed |
|---|---|---|---|---|---|---|---|
| elephant | 0.00 | 0.00 | 0.00 | **0.44** | 0.00 | **0.33** | 2/6 |
| snake | 0.00 | 0.00 | 0.00 | **0.33** | 0.00 | **0.22** | 2/6 |
| tiger | 0.00 | 0.00 | 0.00 | **0.33** | 0.00 | **0.33** | 2/6 |
| wolf | 0.00 | 0.00 | 0.00 | **0.44** | 0.00 | **0.33** | 2/6 |

(S-condition realization rate; full table with P and Δ per model in
`pilot/text/q1_entity_table.md`.)

Elephant was expected to be weak in 4a and 5. It is not weak in 4a — 4a is the family
where *every* entity fails hardest. The variance that Table 1 needs lives in the family
and the modality, not in the entity.

**Consequence for Table 1.** "One entity, six failures" cannot be assembled from the text
modality at this item difficulty: four of the six cells are empty for every entity tested.
Three options, in the order I would try them:

1. **Make Table 1 cross-modal** — one entity, six failures, each cell labelled with the
   modality it was observed in. This works today for five cells with elephant, costs
   nothing conceptually (the paper is a cross-modal benchmark), and turns the modality
   asymmetry from a caveat into the table's own argument.
2. **Fix family 5's construction** before concluding anything about it. One prompt form
   was tested per modality; the negative result is about that form, not about the family.
3. **Raise item difficulty** and re-run the text side. The pilot items are single-clause
   and put the licensing device in the same sentence as the entity — the easiest form. The
   adversarial construction loop (DECISIONS.md #21) exists precisely to find harder items;
   it has not been run.

Two entities, as GOAL.md anticipated, would not help: within a modality the entities do
not differ, and the cell they all miss is the same one.

`pilot/table1_candidates.md` holds the verbatim generations and image paths available
today, per entity and family, with empty cells marked as empty.

---

## 2. Text results (Phase 0)

Realization rate under each condition, and Δ = P − S. Affirmative-realization scoring:
a mention counts only when it is *not* inside the scope of the family's marking device.

| family | model | S | P | A | Δ | S (surface) |
|---|---|---|---|---|---|---|
| 1 existence | llama3.1-8b | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 |
| | qwen3-8b | 0.00 | 0.92 | 0.00 | 0.92 | 0.08 |
| | qwen3-32b | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 |
| 2 attribution | llama3.1-8b | 0.00 | 0.08 | 0.00 | 0.08 | 0.00 |
| | qwen3-8b | 0.00 | 0.58 | 0.00 | 0.58 | 0.25 |
| | qwen3-32b | 0.00 | 0.92 | 0.00 | 0.92 | 0.33 |
| 3 figurative | llama3.1-8b | 0.00 | 0.50 | 0.00 | 0.50 | 0.25 |
| | qwen3-8b | 0.00 | 0.83 | 0.00 | 0.83 | 0.17 |
| | qwen3-32b | 0.00 | 0.83 | 0.00 | 0.83 | 0.42 |
| **4a occlusion** | llama3.1-8b | 0.00 | 1.00 | 0.00 | 1.00 | 0.08 |
| | qwen3-8b | **1.00** | 1.00 | 0.00 | **0.00** | 1.00 |
| | qwen3-32b | **0.17** | 1.00 | 0.00 | 0.83 | 0.75 |
| **4b legibility** | llama3.1-8b | **0.17** | 0.67 | 0.00 | 0.50 | 0.17 |
| | qwen3-8b | **0.50** | 1.00 | 0.00 | 0.50 | 0.58 |
| | qwen3-32b | **0.83** | 1.00 | 0.00 | **0.17** | 0.83 |
| 5 use-mention | llama3.1-8b | 0.00 | 0.83 | 0.00 | 0.83 | 1.00 |
| | qwen3-8b | 0.00 | 1.00 | 0.00 | 1.00 | 1.00 |
| | qwen3-32b | 0.00 | 1.00 | 0.00 | 1.00 | 1.00 |
| **6 relevance** | llama3.1-8b | **0.25** | 0.92 | 0.00 | 0.67 | 0.25 |
| | qwen3-8b | **0.33** | 1.00 | 0.00 | 0.67 | 0.33 |
| | qwen3-32b | **0.33** | 1.00 | 0.00 | 0.67 | 0.33 |

n = 12 items per cell. Every S-condition failure (43 of them) was read individually;
all 43 are genuine. Four things are worth taking out of this table.

**Text over-realization is confined to families 4a, 4b and 6.** The four explicitly
marked families — negation, attribution, simile, word-mention — are at the floor for
every model and every entity. Marking that is *lexically explicit* is respected; marking
that requires tracking a viewpoint (4a, 4b) or inferring relevance (6) is not.

**Family 4a has the largest model gradient in the pilot**: Qwen3-8B fails every single
item (1.00), Qwen3-32B fails 0.17, Llama-3.1-8B fails none. Qwen3-8B's Δ is exactly 0 —
the licensing device is not merely weakened, it is invisible. A representative failure,
verbatim, where the prompt said the elephant was *completely hidden*:

> Tomas can see the top of the stone wall where the elephant's head might be peeking
> over. He notices the shadow of the elephant stretching along the ground.

**Family 4b is the most consistent failure**, and it gets worse with scale: 0.17 → 0.50 →
0.83 across Llama-8B, Qwen3-8B, Qwen3-32B. The stronger model is more willing to report
what the observer cannot see:

> He can see Anna sitting at the table, focused on her laptop. The screen displays the
> message "the meeting is cancelled" in clear view.

**Condition A is at 0.00 in all 21 cells**, so no (entity, scenario) pair was dropped by
rule 3, and the S rates are not inflated by a coincidental base rate. The A generations
were checked for degeneracy (mean 340 characters, none empty).

**The P condition is not at ceiling, and for one cell it collapses.** Llama-3.1-8B
realizes the licensed entity in only 0.08 of family-2 P items — it describes the ridge
and simply drops the elephant that the context asserted. Δ for that cell is therefore
uninformative rather than small. This is a task-design problem, not a model finding: a
description request does not oblige the model to mention everything in the context. The
benchmark needs either an explicit "include everything present" instruction in the P
condition or a task where omission is not licensed.

### Q3 — is family 3 at ceiling in text?

**Yes.** Literal realization is 0.00 for all three models. The surface rate is nonzero
(0.17–0.42) but every one of those mentions is a correctly marked figure of speech —
`wolf-like hunger`, `a wolf's hunger in his gaze`, `the ferocity of a wolf`, `the
deliberate grace of a heavyset elephant`. They are listed in
`pilot/text/surface_vs_affirmative_disagreements.jsonl`.

This is worth flagging as a methodological hazard, not only a result: under GOAL.md's
specified surface-match scoring, family 3 would have shown an S rate of up to 0.42 and
looked like the second-worst family in the pilot. It is in fact the best. Any figurative
scoring that is not scope-aware will manufacture an effect that is not there.

**But the image side reverses it.** The same items rendered by FLUX.1-dev produce literal
realization in 0.58 of S images (§3) — *the porter arriving at the inn, as heavy as an
elephant* returns an elephant walking down the street. Family 3 is therefore not a weak
family; it is **the cleanest modality asymmetry in the pilot**: 0.00 in text, 0.58 in
images, on identical stimuli. That is a stronger result than a full text-side family would
have been, and it is exactly the outcome GOAL.md called a legitimate finding.

Recommendation: keep family 3, but make its claim the asymmetry rather than a rate, and
give it the anthropomorphic-blend observation from §3 as its own scoring category.

---

## 3. Image results (Phase 2)

FLUX.1-dev, 1024x1024, 50 steps, guidance 3.5. S, P and A share a seed per item, so the
three images of an item differ only by the licensing device.

| family | S | P | A | Δ | scored by |
|---|---|---|---|---|---|
| 1 existence | 0.17 | 0.92 | 0.00 | +0.75 | judge |
| 2 attribution | 0.67 | 0.83 | 0.00 | +0.17 | **inspection** (judge: 0.50 / 0.67) |
| 3 figurative | 0.58 | 1.00 | 0.00 | +0.42 | **inspection** (judge: 0.08 / 0.25) |
| **4a occlusion** | **1.00** | 0.83 | 0.00 | **−0.17** | judge |
| **4b legibility** | **0.83** | 0.92 | 0.00 | **+0.08** | **inspection** (judge: 0.42 / 0.75) |
| 5 use-mention | 0.00 | 1.00 | 0.00 | +1.00 | judge |
| **6 relevance** | **1.00** | 0.92 | 0.00 | **−0.08** | judge |

n = 12 items per cell. Families 2, 3 and 4b are reported from direct inspection because
the judge is not reliable on them (§4); families 1, 4a, 5 and 6 are plain presence checks
and the judge's calls were spot-checked against the contact sheets and agreed.

**The image modality fails where the text modality does not.** Negation, attribution and
simile are at the floor in text and well above it in images. The clearest single case, in
a family that text solves perfectly:

> `a hotel lobby at midnight with no elephant in it` → an elephant standing in the
> doorway of the lobby (`pilot/images/1_existence/1_existence_00_S.png`)

**Families 4a and 6 have Δ ≤ 0 in images.** Both sit at S = 1.00: the elephant is in
plain view in front of the wall it was said to be *completely hidden* behind, in all 12
items; the entity mentioned in an explicitly unrelated aside appears in all 12 relevance
items. The licensing device is not weakened, it is inert. A negative Δ is not a
suppression effect in reverse — it is S and P both saturated, with P losing a couple of
items to ordinary generation noise.

**Family 3 realizes the simile in a way the taxonomy does not have a slot for.** Three of
the seven figurative failures are neither "a literal tiger in the scene" nor "an
explicitly marked figure of speech" but an **anthropomorphic blend** — a tiger-headed man
in a suit for *the auditor, as fierce as a tiger*
(`pilot/images/3_figurative/3_figurative_08_S.png`), a wolf-headed man for *as hungry as a
wolf*. The vehicle is fused into the tenor rather than placed beside it. The marking-vs-
binary dichotomy (DECISIONS.md #12) does not cover this, and both the judge and a naive
presence check score it wrongly. It should become a third scoring category, or family 3's
question must name it explicitly.

**Family 5 did not reproduce the failure the taxonomy predicts, in either direction.**
`a crate of elephants` gives crates of elephants and no lettering; `a crate stencilled with
the word ELEPHANT` gives a crate with ELEPHANT on it, legibly, and no animal. Δ = +1.00.
This is the Table 1 example in GOAL.md, and FLUX.1-dev simply does not make that mistake
at this prompt form. Either the typographic failure needs a different construction (a
weaker T2I model, a brand-like or label-like context, or a mention embedded in a longer
prompt), or family 5 is not a live cell for this model.

*Secondary question — does FLUX render text legibly enough for OCR?* **Yes.** The P
condition produced clean, readable SNAKE / TIGER / WOLF stencils, and family 4b's
inscriptions were readable enough to score by eye. Two of twelve 4b P images corrupted the
string ("flight 402 at **sine**", "at **tine**"), so an OCR-based scorer needs edit-distance
matching rather than exact match.

**One stimulus bug and one design bug, both caught by condition A.**

1. *Stimulus:* the person name **Iris** is rendered as an iris flower. Every
   `2_attribution` orchard item returns a close-up of an iris — the A images for items
   02, 05, 08 and 11 are flowers. Person names in image prompts must be screened against
   common nouns.
2. *Design:* family 6's neutral prior remark still contains "wildlife park", which is a
   strong semantic associate of the animal entities. Two A images (`6_relevance_09_A`,
   `6_relevance_11_A`) contain a canid and a leopard. The neutral context must not name an
   animal-bearing location at all. This did not distort the reported numbers — the judge
   asks about the specific entity — but at benchmark scale it would.

### 3.1 Family 5 retested with text-bearing carriers

The crate construction gave FLUX no reason to render text at all, so its null result was
about the stimulus, not the family. Retested with entities that *must* carry text — book,
letter, newspaper, menu — where the norm is that the object shows its content, not its own
category name (a cover shows a title, not the word BOOK). 12 items x S/P/A, all 36 images
inspected by hand.

| condition | word form rendered |
|---|---|
| S (`a wooden desk by a window with a book on it`) | **1/12** (and that one marginal: a handwritten card headed "Letter") |
| P (`…the word BOOK printed across it`) | 12/12 |
| A (scenario only) | 0/12 |

Δ ≈ +0.92 — the same verdict as the crate version, **but for a different and more
informative reason.** The carriers did force text to be rendered; what they did not force
is a *lexical commitment*. FLUX writes pseudo-text: blank sheets for two of the three
letters, unreadable grey page texture for the books, garbled mastheads for the newspapers
(`Tnark Times`, `The Keltic Arples`), garbled headings for the menus (`NERU KME`, `Serus`).
It neither writes BOOK nor writes a title. The P condition proves the capability is there —
told which word to print, it prints it cleanly.

So the image-side use–mention failure needs a generator that **spontaneously writes real
words**. FLUX.1-dev does not, at any prompt form tried here, which makes it the wrong
instrument for family 5 rather than family 5 being the wrong family. A model with much
stronger text rendering (Qwen-Image is Apache-2.0 and runs locally) is the thing to try
before writing this cell off. Note the near-miss: `NERU KME` is a few characters from MENU,
so the pull toward the category word may be there, below the model's rendering fidelity.

### 3.2 Family 4a rebuilt, with an explicitness ladder

The first 4a construction was ambiguous in three ways: the viewpoint anchor was vacuous
(the observer was never in frame and the camera was not his), a correctly suppressed S
image was indistinguishable from A, and a failure could not be told apart from the model
simply ignoring the barrier. v2 puts the camera *at* the observer, and splits S into two
strengths of marking:

- **S_exp** — states the perceptual fact: `the view from where Tomas stands: a high stone wall, with an elephant behind it that Tomas cannot see`
- **S_imp** — gives only the spatial relation: `…with an elephant on the far side`

| condition | entity visible | Δ |
|---|---|---|
| S_exp | **0.83** (10/12) | +0.17 |
| S_imp | **0.67** (8/12) | +0.33 |
| P | 1.00 | |
| A | 0.00 | |

**The construction is better.** v1 sat at 1.00 with Δ = −0.17, saturated and uninformative;
v2 leaves headroom and grades the failure — an elephant filling the archway is not the same
as a wolf's head on a distant tower, and v1 could not tell them apart. The observer is now
in frame in most images, the barrier is drawn in all of them (so "ignored the occlusion" is
now separable from "never drew the wall"), and A is clean at 0.00.

**A direction worth testing, not a finding: explicit marking failed *more* than implicit.**
Saying "that Tomas cannot see" made the animal more likely to appear, and more prominent.
That is the ironic-process / white-bear pattern, and DECISIONS.md §1 already notes that the
white-bear analogy is taken in the negation literature. But the difference is 10 items
against 8, n = 12 — squarely inside noise. It is a hypothesis the explicitness ladder should
be designed to test across families, not a result.

**Generalisation worth keeping.** The implicit/explicit contrast is a within-family version
of the ordering the whole taxonomy rests on — GOAL.md orders the six families "by how
explicitly the input marks the suppression". Building that contrast into every family turns
a stipulated ordering into a measured one, which is a cheap way to make the taxonomy
load-bearing rather than descriptive. It also gives the dataset the implicit/explicit noise
split that family 6 needs.

---

## 4. Q2 — can a VLM judge score families 2 and 4b?

**Short answer, after §4.3: yes, all three — but only once the questions are rewritten.
The judge was never the problem; the question format was.** The multiple-choice protocol
this pilot started with reaches kappa 0.80 / 0.42 / 0.23 on families 2 / 3 / 4b. A single
plain binary question reaches **0.94 / 1.00 / 0.94** on exactly the same images.

The sections below are kept in the order the work happened, because the wrong turns are
the evidence for the diagnosis: §4 diagnoses the multiple-choice failure, §4.1 shows the
paper's paired protocol only half-fixes it, and §4.2–4.3 find the four wording faults that
caused all of it.

All 36 images of families 2, 3 and 4b were inspected directly, blind to the judge's
answers, using the same option set the judge saw.

| family | n | raw agreement | Cohen's κ | disagreements |
|---|---|---|---|---|
| 2 attribution | 36 | **0.89** | **0.80** | 4 |
| 3 figurative | 36 | 0.58 | 0.42 | 15 |
| **4b legibility** | 36 | **0.42** | **0.23** | 21 |

**Family 2 is reliable, but partly for an uninteresting reason.** FLUX never produced the
"explicitly embedded" rendering that option B describes — no thought bubbles, no dream
haze, no framed insets. The attribution family therefore collapses to a presence check in
practice, and presence is what the judge is good at. The reliability figure should not be
read as evidence that the judge can tell marked from unmarked realization; it has not been
tested on a single positive instance of marking.

**Family 4b fails.** κ = 0.23 is barely above chance. The failure is systematic, not
noisy: the judge answers **B ("the text faces the person, away from the viewer")** for
images where the inscription is squarely facing the camera and fully readable — it appears
to answer from the *situation described* (a person reading something) rather than from
the *geometry rendered*. Two examples where the judge said B and the text is plainly
readable in the image: `4b_legibility_06_S` (a laptop screen turned to the camera while
the woman types side-on — geometrically impossible) and `4b_legibility_07_S`.

The label distributions localise both failures precisely:

| family | judge chose | inspection chose |
|---|---|---|
| 2 attribution | A 14, B **4**, C 18 | A 18, B 0, C 18 |
| 3 figurative | A 4, B **15**, C 17 | A 19, B 0, C 17 |
| 4b legibility | A 14, B 22, C **0** | A 21, B 1, C **14** |

Both judge failures are option-B over-use: given a choice between "really there" and
"there only figuratively / only as an image", the judge takes the hedged option, in 4 of
36 family-2 images and 15 of 36 family-3 images, where a human sees no embedding device at
all. Family 4b adds a second, separate defect: **the judge never once used option C**
("no legible writing"), while inspection used it 14 times — it always commits to a
facing direction, even for a dark laptop lid or a page with nothing written on it. Those
14 forced answers are two-thirds of its disagreements.

**How much this costs depends on what the judge is for, and it is not the final scorer.**
The VLM judge's place in the design is inside the construction loop (`overrealization.tex`
P5): it scores candidate items so the difficulty loop can select and mutate them. Final
evaluation is a separate question, and the benchmark's released numbers rest on annotated
subsets with reported agreement, not on this judge. That downgrades the severity below what
§4's numbers suggest on their own — but it does not make it free, and the consequence is a
different one. A judge that systematically *understates* over-realization will read genuinely
failing items as passing, so the loop will keep mutating items that are already hard enough
and will select for items that confuse the judge rather than items that defeat the generator.
The bias lands on item selection instead of on the headline rate, which is harder to notice
later.

Consequences, in order of how much they cost:

1. **The paper's headline example is at risk, as anticipated** (DECISIONS.md open risk 1).
   Family 4b cannot be scored by this judge at this prompt form.
2. **Family 3 is unreliable under this protocol.** The judge answered "B — only a picture,
   statue, logo, costume or shadow" for images containing a literal elephant walking down a
   street. Its S rate of 0.08 against an inspected 0.58 would have made figurative look
   like the safest family in the image modality when it is among the worst. §4.1 shows
   this is fixable by pairing.
3. **The judge understates over-realization wherever it is wrong.** In all three families
   the judge's S rate is *below* inspection (0.50 vs 0.67; 0.08 vs 0.58; 0.42 vs 0.83). A
   judge-only benchmark would report the models as better behaved than they are.

### 4.1 The paired forced-choice protocol (the paper's primary one)

`overrealization.tex` P2 designates **paired forced choice** as the primary protocol —
show the judge the S and P outputs together, ask which came from which prompt — precisely
because pairing neutralises a judge biased toward one answer. Since option-B over-use is
exactly such a bias, the three families were re-judged that way: S and P tiled side by
side, left/right assigned by a stable hash of the item id, one two-alternative question
(`scripts/judge_paired.py`, results in `pilot/images/paired_forced_choice.json`).

| family | accuracy, all 12 pairs | accuracy, discriminable pairs | left-side choice rate |
|---|---|---|---|
| 2 attribution | 0.75 | **1.00** (n=2) | 0.58 |
| 3 figurative | 0.67 | **1.00** (n=5) | 0.67 |
| 4b legibility | 0.42 | 0.67 (n=3) | **0.08** |

"Discriminable" means direct inspection scored the item's S and P images differently; when
the model over-realizes under S the two images genuinely match and no answer can be right.

**Pairing rescues families 2 and 3.** On every pair that actually differs, the paired
judge is correct — 2/2 and 5/5. Family 3's absolute-protocol κ of 0.42 is therefore a
protocol artefact, not a limit on the model: the same Qwen2.5-VL that called a literal
street elephant "only a picture or statue" identifies it correctly when it has the
licensed image beside it for comparison.

**Pairing does not rescue 4b, and shows why.** The judge chose the right-hand image in 11
of 12 pairs. Its answers are positional, not visual, so the 0.42 is not a weak signal but
close to no signal. Both protocols agree that this family cannot currently be scored
automatically.

**But paired forced choice cannot be the primary protocol for this construct.** It measures
discriminability, not realization, and it degenerates exactly where the phenomenon is:
9 of 12 family-2 pairs, 7 of 12 family-3 pairs and 9 of 12 family-4b pairs were
*indistinguishable* because the model over-realized under S, and those pairs carry no
information at all. The more a model fails, the fewer scorable pairs remain — a benchmark
whose headline metric vanishes as the effect grows. The workable combination is the one
this pilot backed into: absolute per-image judgement for the rate, with paired forced
choice as a validity check on the judge. That is worth a sentence in P2, which currently
argues for pairing on reliability grounds alone.

### 4.2 Five protocols for 4b, and one that works

Five question forms were run over the same 36 hand-inspected family-4b images. All are
reduced to the same decision — *the viewer can read the writing*, which is the
over-realization — so the baseline is re-scored as a binary too and the numbers are
comparable. Ground truth: the viewer can read it in 0.58 of the images.

| variant | accuracy | kappa | says "readable" |
|---|---|---|---|
| v0 the original three-way question | 0.81 | 0.63 | 0.39 |
| v1 two binaries: legible at all, then facing | 0.89 | 0.78 | 0.47 |
| **v2 one binary: can *you* read the writing** | **0.97** | **0.94** | 0.56 |
| v3 the same binary, polarity reversed | 0.69 | 0.30 | **0.89** |
| v4 the three-way question, reasoning allowed first | 0.92 | 0.83 | 0.61 |

The winning question is one sentence: *"Can you, the viewer of this image, read written
words on the letter, screen or page that the person is holding or looking at?"*

v3 is the diagnostic. It asks the identical question with the polarity flipped — *is the
writing turned away from the camera, so that you cannot read it?* — and the judge answers
**yes 89% of the time against a 42% truth rate**. It agrees with whatever the question
presupposes. That single fact explains every anomaly in §4: the option-B over-use, the
never-used option C, and the paired protocol's positional answering.

### 4.3 The same fix on families 2 and 3, and four wording faults

Repeating the exercise on the 72 inspected images of families 2 and 3 produced a cleaner
result than expected: **the marking question is unnecessary**. Asking only *"Is there an
elephant anywhere in this image?"* gives

| family | multiple-choice kappa | plain presence question kappa |
|---|---|---|
| 2 attribution | 0.80 | **0.94** |
| 3 figurative | 0.42 | **1.00** |

FLUX never produced a marked realization in either family — no thought bubbles, no framed
insets, no statues — so presence *is* the whole judgement, and adding a second question
about embedding only creates room to go wrong. If a generator that does produce marked
realizations is ever tested, the second question comes back; on this evidence it should be
added only when the first question's failures show it is needed.

Four faults were isolated, each with its own evidence, and they generalise beyond this
benchmark:

1. **A third "hedged" option gives an uncertain judge somewhere to park.** Removing it is
   most of the gain in every family.
2. **Listing the embedding devices makes the judge answer about the medium.** Asked whether
   the tiger is "a picture, statue, logo, pattern, costume or shadow", it replied *Picture*
   or *Statue* — not even yes or no — for 15 of 19 images containing a live animal. Every
   generated image is a picture, so the option is always true under one reading.
3. **The words "real" and "real live" make it answer about ontological status.** Asked
   *"Is there a real live elephant standing in the scene in this image?"* it answered **No
   on all 72 images**, including ones dominated by an elephant. There is indeed no real
   live elephant: it is a picture of one.
4. **Negatively framed questions are agreed with** (v3 above).

The rule that survives all four: **ask one positive question about depicted content, in
the plainest possible words, with no options and no hedge.**

**This is an in-sample fix.** The questions were chosen after seeing where the originals
failed, on the same 108 images used to score them. They are validated out-of-sample in
§4.4 on images generated afterwards.

`scripts/judge_images.py` now defaults to `--protocol binary`; the multiple-choice
questions remain available as `--protocol multichoice` so both versions can be reported
(GOAL.md rule 4).
Until one works, families 3, 4b — and family 2 whenever a marked instance actually
occurs — need human annotation.

### 4.4 Out-of-sample validation of the corrected protocol

The binary questions were chosen on families 2, 3 and 4b. Families 4a2 and 5b were built
afterwards, and their images played no part in choosing any question. Re-judging everything
under `--protocol binary` and comparing with direct inspection:

| family | kappa | in or out of sample |
|---|---|---|
| 2 attribution | 0.94 | in |
| 3 figurative | 1.00 | in |
| 4b legibility | 0.94 | in |
| **4a2 occlusion v2** | **0.955** | **out** |
| **5b text-bearing** | **1.00** | **out** |

The fix holds on families it was not tuned on. The Q2 answer is therefore: **a 7B open VLM
scores every family in this pilot at kappa ≥ 0.94, provided the question is a single
positive binary about depicted content.** What the pilot originally reported as a judge
limitation was a prompt-engineering failure on my part.

Image-modality rates under the corrected protocol, for the record:

| family | S | S_exp | S_imp | P | A |
|---|---|---|---|---|---|
| 1 existence | 0.17 | | | 0.83 | 0.00 |
| 2 attribution | 0.58 | | | 0.83 | 0.00 |
| 3 figurative | 0.58 | | | 1.00 | 0.00 |
| 4a occlusion (v1) | 1.00 | | | 0.92 | 0.00 |
| 4a2 occlusion (v2) | | 0.83 | 0.75 | 1.00 | 0.00 |
| 4b legibility | 0.75 | | | 0.92 | 0.00 |
| 5 use-mention | 0.00 | | | 1.00 | 0.00 |
| 5b text-bearing | 0.08 | | | 1.00 | 0.00 |
| 6 relevance (v1) | 1.00 | | | 0.92 | 0.00 |

---

## 4.5 Overnight extension (2026-07-27)

Four things were rebuilt or added after the first pass. Two produced results stronger than
anything in the original pilot; one produced a clean negative; one is new taxonomy.

> **Superseded in part by §4.6.** The effect reported in A below is real but roughly half
> the size stated here: v2's implicit condition did not entail occlusion, so some correct
> behaviour was scored as failure. Read §4.6 before quoting any number from A.

### A. The explicitness ladder gives family 4a its largest effect — in text

4a v2 splits S into a stated perceptual fact (S_exp, *"an elephant on the far side, where
Tomas cannot see it"*) and a bare spatial relation (S_imp, *"an elephant on the far side"*).
In text, over 12 items x 3 models:

| model | S_exp | S_imp | P | A |
|---|---|---|---|---|
| llama-3.1-8b | 0.00 | 0.42 | 1.00 | 0.00 |
| qwen3-8b | 0.00 | 0.33 | 1.00 | 0.00 |
| qwen3-32b | 0.00 | 0.67 | 1.00 | 0.00 |

(Corrected 2026-07-29 — the figures first written here were 0.92 / 1.00 / 1.00 for S_imp.
`score_text.py` looked its scope cues up by family name with a fallback, and the rebuilt
cells were never registered, so 4a2 and 4a3 were scored with negation cues only and no
occlusion cues at all. *"but she cannot see the elephant"* — a correct suppression —
counted as a failure. The scorer now raises on an unregistered family instead of falling
back.)

One clause — *where Tomas cannot see it* — still separates near-perfect suppression from
partial failure, but the gap is 0.47 on average, not the ~0.97 first reported.

This is the largest and best-separated effect in the pilot, and it is the same boundary
§2 drew between the families: **what is marked lexically is respected; what has to be
inferred from a spatial or perspectival relation is not.** It also converts the taxonomy's
organising principle from a stipulation into a measurement — GOAL.md orders the six
families "by how explicitly the input marks the suppression", and here that ordering is
reproduced *inside* one family, where entity, scenario and task are held constant.

In the image modality the same contrast runs the other way (S_exp 0.83, S_imp 0.67, §3.2),
which is a difference of 2 items out of 12 and should be treated as noise until it is
powered. The text effect is not noise.

### B. Family 6: two rebuilds, one negative and one that works

The pilot's family 6 marked the irrelevant mention with the words "Unrelated aside:",
making a family defined by the *absence* of marking into a marked one. Two replacements
were built, both 60 items x 4 conditions, both with the entity carried by an ordinary
remark rather than a flagged one.

**v2, transcript form** (`User: … Assistant: … User: By the way, …`) — **fails in the image
modality, and the A condition is what caught it.** FLUX reads the transcript as a
description of a comic strip and draws one: panels, speech balloons and garbled lettering,
in A as much as in S, with the requested scene often missing entirely. The manipulation
became the template. 54 images were generated before the run was stopped; they are kept as
the evidence.

**v3, prose form** (*"My cousin started a new job last month. She looks after an elephant —
apparently it recognises her footsteps now. Anyway — a school gymnasium after the game."*)
— **works.** The requested scene is rendered correctly, no comic-strip artefacts, and the
A condition is clean.

The lesson is narrow but worth recording: a text-to-image model has no context/request
distinction, so any *format* cue in the prompt becomes scene content. Relevance can still
be tested — the structure survives in prose — but not by showing the model a transcript.

**In images, v3 turns a saturated cell into a measurable one** (60 items x 4 conditions,
240 images, corrected binary protocol):

| construction | S | S_exp | S_imp | P | A | Δ |
|---|---|---|---|---|---|---|
| v1 (marked, n=36) | **1.00** | | | 0.92 | 0.00 | **−0.08** |
| v3 (prose, n=240) | | 0.55 | 0.60 | 0.82 | 0.00 | **+0.25** |

v1 sat at the ceiling with a negative Δ and could not discriminate anything. v3 leaves
headroom in both directions and produces a real licence-sensitivity number. Spot-checking
the judge against images inspected by hand before the judge ran: 8 of 8 unambiguous cells
agree; the one disagreement is an item where FLUX drew something between a wolf and a
German Shepherd and the judge took the stricter reading — a stimulus ambiguity, not a
judge error.

Note P = 0.82, not 1.00: the model sometimes omits the entity even when the request names
it. Family 6's P condition needs the same attention §2 flagged for family 2's.

**In text, the better construction produces less over-realization, not more:**

| construction | S | S_imp | S_exp | P |
|---|---|---|---|---|
| v1 (marked "Unrelated aside", n=12) | 0.25–0.33 | — | — | 0.83–1.00 |
| v2 (transcript, n=60) | — | 0.00 | 0.00 | 0.63–0.88 |
| v3 (prose, n=60) | — | 0.00–0.02 | 0.00 | 0.68–0.93 |

**This is confounded and should not be read as "the template fixed it".** v1 and v3 differ
in two ways at once: the marker, and the openness of the request. v1 asked for *a peaceful
place*; v3 asks for *a school gymnasium after the game*, *a harbour at first light*. That
is the openness ladder (P4) appearing as an uncontrolled variable. Openness has to be held
fixed before the constructions can be compared — which is an argument for building the
ladder in from the start rather than adding it later.

### C. Oblique realization — a third outcome the taxonomy has no slot for

The family 6 v3 text generations contain 10 surface hits under S_imp. Nine of them are not
the entity:

> A child's **drawing of a tiger** hangs on the wall
> a faded **mural of a wolf** on the wall
> a **calendar with a tiger** on each month
> the distant howl of a **wolf on the calendar**
> a hastily scrawled **elephant drawing**

The entity is neither absent nor present as a referent. It is smuggled in as a **depiction
inside the scene**, on a carrier the scene independently licenses — a waiting room may have
a calendar. This is the same move as family 3's anthropomorphic blends (§3), and the same
distinction that broke the VLM judge, which kept answering "it is only a picture" (§4.3).

`scripts/score_text.py` now reports it as its own flag, `realized_oblique`, alongside
surface and affirmative realization.

**The control that makes it interesting: oblique realization happens under P too**, at
0.08–0.43 across models in families 6b and 6c. When the request *licenses* the tiger, the
model still often renders it as a poster. So obliqueness is not a suppression strategy —
it is how the model reconciles a mentioned entity with a scene the entity does not fit.
Under S it looks like leakage; under P it looks like compliance; the mechanism is the same.

Three consequences:

1. **It is an intrusion by the DRM measure and a success by the referential one.** The
   paper cannot leave this to the annotator's judgement — DECISIONS.md #11 fixes the unit
   of realization as *information made accessible to the audience*, and a drawing of a
   tiger does make the tiger accessible. On that definition it is a failure.
2. **It predicts where the binary/marking dichotomy breaks** (DECISIONS.md #12): a third
   category is needed, and it is needed in at least families 3 and 6, in both modalities.
3. **It is a confound for the P condition.** If P is satisfied obliquely, Δ understates
   licence sensitivity, because the numerator is not measuring the same kind of realization
   as the denominator.

### D. Measuring naturalness: the obvious method does not work

DECISIONS.md #24 calls for naturalness to be measured and reported next to difficulty.
The obvious instrument is per-token negative log-likelihood under a base LM
(`scripts/score_naturalness.py`, Qwen3-8B, no chat template). Two things were learned, both
negative, and both worth recording so the next attempt does not repeat them.

**First, the baseline matters more than the model.** Scoring S against the A condition
measures length, not naturalness: per-token NLL falls as a sequence gets longer, and A is
much shorter than S, so ten of eleven family cells came out "more natural than the
scenario alone". The script now uses **P** as the baseline, which is length-matched to S by
construction (rule 1), and reports token counts alongside so residual mismatch is visible.

**Second, and more seriously: per-token perplexity does not measure what "natural" means
here.** Ranking family 3's items by NLL:

| item | NLL |
|---|---|
| *the auditor entering the office, as hungry as a wolf* — the awkward one | 4.99 (among the **most** natural) |
| *the coach at halftime, as fierce as a tiger* — the idiomatic one | 5.64 (among the **least** natural) |

The score is dominated by the predictability of the scenario phrase — every item using
*the porter arriving at the inn* scores low and every item using *the coach at halftime*
scores high — and is nearly blind to whether the vehicle fits the tenor, which is exactly
the property that makes an item read as written by a person.

What to try instead: a **conditional** measure that isolates the device from the scenario,
e.g. log P(device | scenario) − log P(device | neutral prefix), a PMI-style fit score.
That is one experiment, not a research programme, but it was not run here. Until something
is validated against human judgement, naturalness should be a human-rated field on a
sample rather than an automatic filter — and in particular it should **not** be wired into
the P5 difficulty loop as a floor while the instrument is this weak.

One incidental finding from the same run: in family 3 the S prompts are on average *more*
natural than the P prompts (cost −1.28 nats/token, lengths matched). *As fierce as a tiger*
is idiomatic; *standing beside a tiger* is not. So for the figurative family the
suppression condition is the natural one and the licensed control is the odd one — the
opposite of the usual worry, and a reason to check both conditions rather than only S.

---

## 4.6 Round three (2026-07-28): the wording fix, and the verb test

Two constructions were rebuilt after review. One corrects a validity flaw that had
inflated the pilot's headline effect; the other answers a design question with a clear
negative.

### A. 4a3 — occlusion that is actually entailed

**The flaw.** v2's implicit condition read *"a tall wooden fence, with an elephant on the
far side of it"*. That does not entail invisibility: an elephant is taller than a fence,
so rendering it above the fence is a faithful reading of the prompt. S_imp was a weaker
*fact*, not a weaker *marking* of the same fact — so an unknown share of the "failures"
were correct behaviour.

**The fix.** Every barrier is given an absolute height that exceeds every entity (a
four-metre brick wall, a five-metre flood wall, the windowless side of a warehouse) and
the entity is placed *entirely* beyond it. Occlusion now follows from the geometry alone.
n doubles to 24 items (4 entities x 6 barriers), because this cell carried the largest
claimed effect on only 12.

**Text — the effect survives at about half its reported size, and stops being consistent:**

| model | v2 S_exp → S_imp | gap | v3 S_exp → S_imp | gap |
|---|---|---|---|---|
| llama-3.1-8b | 0.00 → 0.42 | 0.42 | 0.00 → 0.21 | **0.21** |
| qwen3-8b | 0.00 → 0.33 | 0.33 | 0.21 → 0.25 | **0.04** |
| qwen3-32b | 0.00 → 0.67 | 0.67 | 0.00 → 0.17 | **0.17** |
| mean | | **0.47** | | **0.14** |

Two corrections, not one. The wording flaw is real and cuts the gap by about two thirds.
On top of that, every number first written into this section was inflated by the scorer
bug described above: the rebuilt cells were scored without occlusion cues, so correct
suppressions counted as failures. The figures in this table are the corrected ones.

§4.5A called this "the largest and best-separated effect in the pilot". It is neither: the
mean gap is 0.14 after the wording fix, and on qwen3-8b it is 0.04. Text-side occlusion
over-realization is also much rarer than reported — S_imp runs at 0.17–0.25, not 0.71.
The explicitness ladder is still worth building into every family, but as one factor among
several. DECISIONS.md #32 is corrected accordingly.

**Images — the failure is real, and the mechanism is specific:**

| | S_exp | S_imp | P | A |
|---|---|---|---|---|
| v2 (n=48) | 0.83 | 0.75 | 1.00 | 0.00 |
| v3 (n=96) | 0.67 | 0.83 | 1.00 | 0.00 |

With occlusion geometrically entailed, FLUX still renders the entity in two thirds to four
fifths of S images — so the image-side result does *not* depend on the wording flaw. What
the images show is not that the barrier is ignored: **the barrier is drawn in every single
image.** The model instead reconstrues the scene so that the barrier does not occlude —
splitting one wall into two with a gap between them and putting the elephant in the gap
(`4a3_occlusion_v3_00_S_exp`), standing the elephant *on top* of the wall
(`…_00_S_imp`), rendering a four-metre hoarding as a free-standing panel in an open field
(`…_01`), turning "the windowless side of a warehouse" into an alley with the elephant at
the far end (`…_02`).

That is the same move as oblique realization (§4.5C): rather than violate the instruction
outright, the model finds a configuration that looks licensed and still lets the entity be
shown. It suggests the two phenomena are one — **the model satisfies the mention by
finding an admissible frame for it** — which is a better claim than either on its own.

Note also that the S_exp / S_imp direction flips between v2 and v3 in the images (0.83 vs
0.67, then 0.67 vs 0.83). Both differences are 2–4 items. There is no image-side
explicitness effect to report.

### B. 6d — adding "draw" to the request makes things worse

6c's image request ends *"Anyway — a school gymnasium after the game."* with no verb.
Adding one was proposed on the grounds that it would make the request clearer. It does the
opposite, over 60 items x 4 conditions:

| | S_exp | S_imp | P | A | Δ |
|---|---|---|---|---|---|
| 6c (no verb) | 0.55 | 0.60 | 0.82 | 0.00 | **+0.25** |
| 6d (`draw …`) | 0.68 | 0.65 | 0.77 | 0.00 | **+0.11** |

S rises, P falls, and Δ more than halves. The images show why: *"draw a school gymnasium
after the game, including an elephant"* is read as **a drawing of an elephant** — pencils,
sketch paper and a line-art elephant on the page (`6d_relevance_draw_00_P`). The verb
opens the oblique-realization channel in the P condition, which is precisely where it does
most damage, because P is the denominator of Δ. A second cost: the verb shifts the whole
cell's visual register toward illustration, so 6c and 6d are not stylistically comparable.

The good news is narrow but worth recording: the verb did **not** bring back the v2
comic-strip artefact. A is clean at 0.00 and the requested scenes render correctly, so the
transcript failure was about *format cues*, not about instruction-likeness as such.

**Recommendation: keep 6c, no verb.**

| item | measured |
|---|---|
| text, Qwen3-8B | 52 s load, 2.2 s for 252 prompts (0.009 s/prompt) |
| text, Llama-3.1-8B | 57 s load, 2.5 s for 252 prompts (0.010 s/prompt) |
| text, Qwen3-32B | 69 s load, 6.5 s for 252 prompts (0.026 s/prompt) |
| image, FLUX.1-dev | 12.1 s/image at 1024x1024, 50 steps; 33.9 GB peak VRAM; 5.5 s load |
| judge, Qwen2.5-VL-7B | 252 images in 169 s (0.67 s/image) |

Wall-clock for the whole pilot: about 25 minutes of text and judging, and 51 GPU-minutes
of image generation (halved to ~26 minutes by running one FLUX worker per GPU).

Text generation is free at this scale: batched under vLLM, the whole 6,480-prompt
projection costs **under 3 minutes of GPU time per model**, plus about a minute of load.
Model loading dominates, so the budget should be counted in model-loads, not prompts.

The image side is what costs. At 12.2 s/image, the projected 6,480 prompts x 1 image
= **22 GPU-hours on one H100**, or 11 hours on the two GPUs available here. That is
affordable but not free, and it is the number that should drive any decision to cut the
item count. Two levers: 28 steps instead of 50 (FLUX.1-dev is usable there) roughly
halves it, and a batch size above 1 would help further — neither was used here, so 22 h
is an upper bound.

---

## 6. What this pilot does not settle

1. **Item difficulty is a confound for every negative result.** The four families at the
   floor were tested with the easiest possible items: one clause, licensing device and
   entity in the same sentence, no distractors. "Family 1 is solved in text" is only
   supported for *this* difficulty. The adversarial construction loop has not been run.
2. **n = 12 items per cell, 3 per (entity, family)**. Every rate here has a wide
   interval; treat differences under about 0.25 as noise.
3. **Four entities, all large animals.** No non-animal, no artefact, no abstract noun.
   The claim "the entity does not matter" is only established within that class.
4. **The affirmative-realization scorer is a regex-scope heuristic.** It was corrected
   twice during this pilot after reading the outputs it disagreed with (`crate` wrongly
   marking family-5 P items; `wolf-like` and `belief in the wolf` wrongly counted as
   failures). It is good enough for a pilot and not good enough for the benchmark, which
   needs either a model-based scope judgement or human annotation.
5. **One borderline class in family 4a is unresolved**: generations that leak the
   entity's *presence* without its *appearance* — "the rustling of leaves gave away the
   elephant's slow, heavy steps". Under DECISIONS.md #11 (the realized unit is
   information made accessible to the audience) that is a failure; under a
   visibility-only reading it is not. The scorer currently counts it as suppressed. The
   annotation protocol must decide this explicitly.
6. **No human annotation, no inter-annotator agreement.** The direct inspection in §4 is
   one rater — me — and is a sanity check on the judge, not a substitute for the annotated
   subsets the benchmark needs. Several inspection calls were genuinely borderline and are
   recorded with notes in `pilot/images/inspection.jsonl`: a red serpentine cartoon
   creature scored as "not a snake", a phone screen with text too small to be sure of, and
   the anthropomorphic blends, which I scored as realizations.
7. **One T2I model.** Every image-side negative is a statement about FLUX.1-dev at 50
   steps, not about the family. Family 5 makes this concrete: two different constructions
   both returned 0.00, and the reason turned out to be a property of the generator's text
   rendering (§3.1), not of the stimulus or the family.
8. **Two families were rebuilt mid-pilot** (4a and 5) after their first constructions were
   found to be ambiguous or inert. Their v2 numbers come from 12 items each, inspected by
   one rater, and have not been replicated. The v1 numbers are kept in the repository
   rather than deleted, per GOAL.md rule 4.
9. **`Iris` renders as an iris flower** in the 4a v2 items as well as family 2. Every
   scenario that uses that name is contaminated; the name must be replaced before any of
   these items are reused.
10. **Many items are not natural English.** Family 3 was built by full crossing, against
    DECISIONS.md #16's explicit warning, so 12 items include *the auditor entering the
    office, as hungry as a wolf* and *the coach at halftime, as slippery as a snake*.
    Roughly a third of them read as idiomatic. Items that are hard because they are odd
    are weak evidence, and the P5 difficulty loop will make this worse unless naturalness
    is a floor inside it. See DECISIONS.md #24.

---

## 7. Go / no-go per family

| family | text | image | recommendation |
|---|---|---|---|
| 1 existence | floor | 0.17 | **Go, image-only.** Crowded literature (DECISIONS.md §1); the text side adds nothing at this difficulty. |
| 2 attribution | floor | 0.67 | **Go, image-only**, and note that the judge has never seen a marked instance to be tested on. |
| 3 figurative | floor | 0.58 | **Go**, reframed as the modality asymmetry, with the anthropomorphic blend as a third scoring category. Score it with the paired protocol, not the absolute one. |
| 4a occlusion | 0.39 | **0.83 / 0.67** | **Go — strongest cell**, using the v2 construction (§3.2); v1's stimulus is ambiguous and saturated, retire it. Largest model gradient in text. |
| 4b legibility | 0.50 | **0.83** | **Go, both on the phenomenon and on the scoring** — the single binary reaches kappa 0.94 (§4.2). It can be the headline. |
| 5 use-mention | floor | floor | **Hold — but the diagnosis has changed** (§3.1). Two constructions tried; the blocker is that FLUX writes no real words unprompted. Retry on a strong text-rendering generator before deciding. |
| 6 relevance | **floor** | **0.58** | **Go, on the v3 prose construction** (§4.5B). Text is at the floor once the items are natural, images give Δ = +0.25 with headroom. Another modality asymmetry, and the "wildlife park" leak is gone. |

The two families the pilot recommends building on hardest — 4a and 6 — are the two that
DECISIONS.md §1 marks as most open in the literature. The two that are crowded (1) or
collide in wording (3) are also the two the text modality cannot measure at all.
