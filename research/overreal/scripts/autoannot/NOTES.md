# Automated annotation: working notes

Goal: an automatic annotator over OverReal-Det labels (disruptive / silent /
integrated / withheld), validated against the human gold. Model: Claude Opus 5
through `claude -p`, the same judge the LJ metric uses. Dev/test split of the
gold in `data/overreal_v1/auto_split.jsonl`; wording is iterated on dev only.

Target agreement (2026-09-10): kappa at least 0.6 per family, ideally 0.7.
Human-human kappa for reference: cancellation 0.73, attribution 0.51,
figurative 0.65, perspectival 0.72.

## Label conventions confirmed with the author (2026-09-10)

- Perspectival, silent: the target is not in the hiding place the prompt
  names but appears elsewhere in the scene (a second travel pillow on another
  traveller; a bowl of lemons beside the face-down cookbook).
- Perspectival, disruptive: the target is exposed in the hiding place itself:
  a closed cupboard drawn open, an opaque one drawn transparent, the object
  stuffed into the container in an odd, visible way, a print on the wrong
  side.
- Perspectival, integrated: the target is seen only through a device the
  prompt allows: a reflection, a part peeking out, a cutaway, or the picture
  taken from the in-scene reader's own point of view. Whether the point of
  view counts depends on how the prompt is written.
- Figurative: the target replacing the described scene outright (a tiger
  close-up, no coach) is disruptive. A shape resemblance (a dog-shaped cloud)
  is fine while it stays a shape; once the target is rendered with full
  literal detail it is disruptive. The person of the comparison being turned
  into the animal is disruptive.
- "other" is not in the gold (both-other images were excluded), so the
  annotator has no other branch yet. A gate for corrupted or unrelated images
  is to be added for the full run and its false-positive rate checked on gold.

## Dev results (Claude Opus 5, kappa on four labels, dev split)

| protocol | cancellation | attribution | figurative | perspectival |
|---|---|---|---|---|
| v1 | 0.88 | 0.56 | 0.66 | 0.39 |
| v3 | | 0.43 | 0.63 | 0.55 |
| v4 | | 0.43 | | |
| v5 | | **0.59** | 0.63 | |
| v6 | | | | **0.59** |
| v7 | | 0.57 | | 0.49 |

Local Qwen2.5-VL-7B and Qwen3-VL-32B were tried under v1/v2 and dropped
(0.15 to 0.52 per family); the annotator uses the same Claude judge as LJ.

`final` = cancellation v1, attribution v5, figurative v1, perspectival v6,
each behind the Q0 "other" gate (0 false positives on the 122 attribution
gold images under v4, 1 on 469 figurative under v5). Frozen 2026-09-10
before the test-split run. Output files:
`data/overreal_v1/auto/final__claude-opus-5.jsonl` (test split) and
`final__claude-opus-5__sample.jsonl` (eval_sample plus ambiguous).

## Protocol history

- v1: first cascade. Claude on dev: cancellation 0.88, attribution 0.56,
  figurative 0.66, perspectival 0.39 (kappa, four labels).
- v2: attribution Q2 -> contradiction; perspectival rewording. Only run with
  Qwen 7B; superseded by v3 before Claude ran it.
- v3: from Claude v1 errors. Figurative Q1 "in any form"; Q2 covers scene
  replacement; perspectival asks the device question before the exposure
  question; text targets ask whether the written side faces the viewer.
- v4: tree format, Q0 other gate; attribution Q2b "could a real X plausibly
  be there" (sent disruptive to silent; dropped).
- v5: attribution Q2b prominence ("right beside the person"); figurative Q2
  "dominates or replaces the scene" (no gain over v1).
- v6: perspectival exposure question first, naming open / transparent /
  removed / wrong side; device list without "see-through" and "peeking".
- v7: attribution Q2b on the kind of mental state (present belief vs
  memory); perspectival exposure adds "visible side" and "beside the
  container". Neither beat the previous version; tuning stopped here.

## Test-split results (2026-09-11, protocol `final`, frozen before this run)

Kappa on four labels against the human gold, test split (1,118 images).
Opus = claude-opus-5 through the Anthropic API (effort low, adaptive
thinking, image block cached); GPT = gpt-5.2-2025-12-11 through the OpenAI
API. "agreed" = the subset where the two annotators give the same label.

| family | n | Opus | GPT | Opus-GPT | agreed % | Opus on agreed (k4 / k2) |
|---|---|---|---|---|---|---|
| cancellation | 101 | 1.00 | 0.88 | 0.88 | 95 | 1.00 / 1.00 |
| attribution | 117 | 0.42 | 0.47 | 0.67 | 76 | 0.52 / 0.63 |
| figurative | 463 | 0.60 | 0.40 | 0.59 | 72 | 0.58 / 0.73 |
| perspectival | 437 | 0.41 | 0.42 | 0.34 | 64 | 0.63 / 0.65 |
| all | 1,118 | 0.58 (k2 0.70) | 0.50 | | | |

Harness check: on the 317 test images that the `claude -p` run had also
labelled under `final`, CLI and API agree at kappa 0.79 to 0.98 and score the
same against gold (attribution 0.46 vs 0.42, figurative 0.67 vs 0.64), so the
drop from dev is wording overfit on attribution and perspectival, not the
transport. Perspectival errors on test: disruptive -> withheld 17,
silent -> disruptive 24, integrated -> disruptive 20.

Files: `final__claude-opus-5-api.jsonl` (test), `final__claude-opus-5-api__sample.jsonl`
(eval_sample + amb), `final__gpt-52-2025-12-11.jsonl` (test),
`final__gpt-52-2025-12-11__sample.jsonl` (eval_sample + amb).
Spend: Opus test split about 6 USD; GPT test split about 4.9M input tokens.

## Full-sample runs (2026-09-11)

Both annotators cover the eval sample plus the ambiguous split (7,432 images;
3 rows of the sample have no image file). Table 2 block (b) is filled from
the Opus run (`fill_main_table.py --outcomes auto`). Opus spend: about 90 USD
for the sample, 6 USD for the test split.

Opus vs GPT agreement on the sample (four labels / coarse):

| family | n | agree | kappa4 | kappa2 |
|---|---|---|---|---|
| cancellation | 1,700 | 95% | 0.87 | 0.87 |
| attribution | 1,788 | 79% | 0.69 | 0.81 |
| figurative | 2,034 | 78% | 0.64 | 0.65 |
| perspectival | 1,908 | 63% | 0.31 | 0.65 |

"other" fires on 0 to 0.7% of images under Opus and 0.4 to 2.9% under GPT.
