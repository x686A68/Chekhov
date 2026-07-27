# Prompt naturalness

Per-token NLL under `Qwen/Qwen3-8B` (a base LM, not an instruct model — an
instruction-tuned model rates imperative prompts as natural because it was
trained on them). Lower is more natural.

`cost` is the S-condition NLL minus the same item's **P**-condition NLL. P is the
baseline rather than A because S and P are length-matched by construction while A
is much shorter, and per-token NLL falls with length. Positive cost = the
suppression device makes the prompt less natural than the licensed version.

## 1_existence

mean P-condition NLL 5.2642 over 11.7 tokens, mean excess 0.8036

| | cost | prompt |
|---|---|---|
| least natural | +1.38 | a hotel lobby at midnight with no wolf in it |
| least natural | +1.14 | a hotel lobby at midnight with no tiger in it |
| least natural | +1.12 | a hotel lobby at midnight with no snake in it |
| most natural | +0.36 | a riverside campsite at dawn with no elephant in it |
| most natural | +0.47 | a riverside campsite at dawn with no wolf in it |
| most natural | +0.50 | a school gymnasium after the game with no elephant in it |

## 2_attribution

mean P-condition NLL 4.0758 over 15.3 tokens, mean excess 0.2172

| | cost | prompt |
|---|---|---|
| least natural | +0.38 | Tomas at the loading dock, believing that there is a tiger at the loading dock |
| least natural | +0.33 | Tomas at the loading dock, believing that there is an elephant at the loading dock |
| least natural | +0.32 | Tomas at the loading dock, believing that there is a wolf at the loading dock |
| most natural | +0.04 | Maya at the ridge, believing that there is a tiger at the ridge |
| most natural | +0.06 | Maya at the ridge, believing that there is an elephant at the ridge |
| most natural | +0.06 | Maya at the ridge, believing that there is a wolf at the ridge |

## 3_figurative

mean P-condition NLL 6.5398 over 10.0 tokens, mean excess -1.2788

| | cost | prompt |
|---|---|---|
| least natural | -0.82 | the auditor entering the office, as heavy as an elephant |
| least natural | -0.96 | the porter arriving at the inn, as heavy as an elephant |
| least natural | -0.97 | the porter arriving at the inn, as fierce as a tiger |
| most natural | -1.88 | the coach at halftime, as hungry as a wolf |
| most natural | -1.63 | the coach at halftime, as fierce as a tiger |
| most natural | -1.45 | the coach at halftime, as heavy as an elephant |

## 4a2_occlusion_v2

mean P-condition NLL 4.9266 over 18.3 tokens, mean excess -0.5834

| | cost | prompt |
|---|---|---|
| least natural | -0.36 | the view from where Iris stands: a tall wooden fence, with a wolf behind it that Iris cannot see |
| least natural | -0.39 | the view from where Tomas stands: a high stone wall, with a wolf behind it that Tomas cannot see |
| least natural | -0.42 | the view from where Tomas stands: a high stone wall, with an elephant behind it that Tomas cannot see |
| most natural | -0.84 | the view from where Delia stands: a parked lorry, with a tiger behind it that Delia cannot see |
| most natural | -0.78 | the view from where Delia stands: a parked lorry, with a snake behind it that Delia cannot see |
| most natural | -0.70 | the view from where Iris stands: a tall wooden fence, with a snake behind it that Iris cannot see |

## 4a_occlusion

mean P-condition NLL 6.4205 over 17.7 tokens, mean excess 0.0682

| | cost | prompt |
|---|---|---|
| least natural | +0.91 | a snake completely hidden behind the parked lorry, photographed from where Iris stands |
| least natural | +0.80 | a snake completely hidden behind the high stone wall, photographed from where Tomas stands |
| least natural | +0.54 | a tiger completely hidden behind the high stone wall, photographed from where Tomas stands |
| most natural | -0.57 | a wolf completely hidden behind the hedge at the end of the yard, photographed from where Delia stands |
| most natural | -0.52 | an elephant completely hidden behind the hedge at the end of the yard, photographed from where Delia stands |
| most natural | -0.29 | a tiger completely hidden behind the hedge at the end of the yard, photographed from where Delia stands |

## 4b_legibility

mean P-condition NLL 5.4393 over 18.4 tokens, mean excess -0.9251

| | cost | prompt |
|---|---|---|
| least natural | -0.36 | Anna sits at the table reading a phone screen that says "the meeting is cancelled" |
| least natural | -0.53 | Anna sits at the table reading a phone screen that says "I'm sorry" |
| least natural | -0.71 | Anna sits at the table reading a phone screen that says "flight 402 at nine" |
| most natural | -1.30 | Anna sits at the table reading a letter that says "the meeting is cancelled" |
| most natural | -1.26 | Anna sits at the table reading a notebook page that says "the meeting is cancelled" |
| most natural | -1.18 | Anna sits at the table reading a letter that says "I'm sorry" |

## 5_use_mention

mean P-condition NLL 4.4805 over 14.7 tokens, mean excess 1.5255

| | cost | prompt |
|---|---|---|
| least natural | +2.28 | the warehouse with a crate of elephants |
| least natural | +1.96 | the freight yard with a crate of elephants |
| least natural | +1.75 | the warehouse with a crate of snakes |
| most natural | +1.09 | the loading bay with a crate of tigers |
| most natural | +1.14 | the freight yard with a crate of tigers |
| most natural | +1.21 | the warehouse with a crate of tigers |

## 5b_text_bearing

mean P-condition NLL 5.0949 over 17.5 tokens, mean excess 0.845

| | cost | prompt |
|---|---|---|
| least natural | +2.91 | a shop counter with a newspaper on it |
| least natural | +1.70 | a shop counter with a menu on it |
| least natural | +1.61 | a shop counter with a letter on it |
| most natural | -0.20 | a wooden desk by a window with a book on it |
| most natural | -0.09 | a wooden desk by a window with a letter on it |
| most natural | +0.20 | a cafe table in the morning with a book on it |

## 6_relevance

mean P-condition NLL 5.1879 over 23.3 tokens, mean excess -0.0685

| | cost | prompt |
|---|---|---|
| least natural | +0.18 | a bedtime story illustration. Unrelated aside: my cousin looks after a wolf at a wildlife park. |
| least natural | +0.17 | a bedtime story illustration. Unrelated aside: my cousin looks after a snake at a wildlife park. |
| least natural | +0.17 | a peaceful place. Unrelated aside: my cousin looks after a snake at a wildlife park. |
| most natural | -0.38 | a busy market. Unrelated aside: my cousin looks after an elephant at a wildlife park. |
| most natural | -0.29 | a peaceful place. Unrelated aside: my cousin looks after an elephant at a wildlife park. |
| most natural | -0.19 | a busy market. Unrelated aside: my cousin looks after a tiger at a wildlife park. |

## 6b_relevance_v2

mean P-condition NLL 4.3986 over 53.1 tokens, mean excess -0.1289

| | cost | prompt |
|---|---|---|
| least natural | +0.26 | User: The pub quiz last night had an entire round on a tiger. We got two out of ten.
Assistant: Two out of ten is at least memorable.
User: By the way, please draw a harbour at first light. |
| least natural | +0.25 | User: The pub quiz last night had an entire round on an elephant. We got two out of ten.
Assistant: Two out of ten is at least memorable.
User: By the way, please draw a school gymnasium after the game. |
| least natural | +0.23 | User: My brother swears he heard an elephant while camping last summer. Nobody else did.
Assistant: Camping does strange things to the ears.
User: By the way, please draw a hospital waiting room. |
| most natural | -0.45 | User: We went to a talk at the museum yesterday. Most of it was about how an elephant finds its way over long distances.
Assistant: That sounds like an interesting evening.
User: By the way, please draw a school gymnasium after the game. |
| most natural | -0.40 | User: My daughter has decided that a wolf is her favourite animal. She draws nothing else at the moment.
Assistant: Children get wonderfully single-minded about these things.
User: By the way, please draw a bakery just before closing. |
| most natural | -0.39 | User: My daughter has decided that a tiger is her favourite animal. She draws nothing else at the moment.
Assistant: Children get wonderfully single-minded about these things.
User: By the way, please draw a hospital waiting room. |

## 6c_relevance_v3

mean P-condition NLL 4.7259 over 34.9 tokens, mean excess -0.1315

| | cost | prompt |
|---|---|---|
| least natural | +0.48 | The pub quiz last night had an entire round on a tiger. We got two out of ten. Anyway — a harbour at first light. |
| least natural | +0.39 | The pub quiz last night had an entire round on a wolf. We got two out of ten. Anyway — a launderette on a rainy evening. |
| least natural | +0.34 | There is a charity near us that raises money for a tiger. They had a stall at the market on Saturday. Anyway — a harbour at first light. |
| most natural | -0.56 | My daughter has decided that a wolf is her favourite animal. She draws nothing else at the moment. Anyway — a bakery just before closing. |
| most natural | -0.51 | I lost an argument at dinner about whether a tiger can recognise itself in a mirror. Anyway — a hospital waiting room. |
| most natural | -0.50 | I could not sleep last night. There was a documentary on about a wolf and I ended up watching the whole thing. Anyway — a hospital waiting room. |
