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

---

## 4. Q2 — can a VLM judge score families 2 and 4b?

**Family 2: yes. Family 4b: no, under either protocol. Family 3, which was not on the
list, fails the absolute protocol but is fully recovered by the paired one (§4.1).**

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

### 4.2 Remaining options for 4b

What to try before giving up on automatic scoring, now that the failure mode is known:
split every three-way question into binaries so the hedge option cannot absorb the
uncertainty — for 4b, *"is any writing legible in this image?"* and only then *"which way
does it face?"*; for 2 and 3, *"is there a real X in the scene?"* and only then *"is it
inside a thought bubble / a picture frame?"*. Cropping to the carrier before asking may
also help 4b. None of these was tested here.
Until one works, families 3, 4b — and family 2 whenever a marked instance actually
occurs — need human annotation.

---

## 5. Cost

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
7. **One T2I model, one prompt form per family.** Every image-side negative — family 5
   above all — is a statement about FLUX.1-dev at 50 steps with this phrasing, not about
   the family.

---

## 7. Go / no-go per family

| family | text | image | recommendation |
|---|---|---|---|
| 1 existence | floor | 0.17 | **Go, image-only.** Crowded literature (DECISIONS.md §1); the text side adds nothing at this difficulty. |
| 2 attribution | floor | 0.67 | **Go, image-only**, and note that the judge has never seen a marked instance to be tested on. |
| 3 figurative | floor | 0.58 | **Go**, reframed as the modality asymmetry, with the anthropomorphic blend as a third scoring category. Score it with the paired protocol, not the absolute one. |
| 4a occlusion | 0.39 | **1.00** | **Go — strongest cell.** Works in both modalities, Δ = 0 in images, largest model gradient in text. |
| 4b legibility | 0.50 | **0.83** | **Go on the phenomenon, no-go on the scoring.** Needs a working protocol before it can be the headline. |
| 5 use-mention | floor | floor | **Hold.** Not observed in either modality; retry with other constructions before deciding. |
| 6 relevance | 0.31 | **1.00** | **Go.** Consistent in text, saturated in images; fix the "wildlife park" leak in the neutral context first. |

The two families the pilot recommends building on hardest — 4a and 6 — are the two that
DECISIONS.md §1 marks as most open in the literature. The two that are crowded (1) or
collide in wording (3) are also the two the text modality cannot measure at all.
