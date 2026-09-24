# Rubric: does a real prompt contain an over-realization situation?

You are given text-to-image prompts written by real users. For each prompt decide
whether it mentions a **target** inside one of four situations, where the prompt
does **not** ask for the target to be shown as an ordinary part of the scene.
Follow the definitions below exactly. Selection is on the **form of the prompt
only**; never guess what a model would draw.

## Global conditions (all must hold for a positive)

1. The target is a concrete, depictable thing: an object, creature, person,
   substance, or a specific string of text. Abstract nouns (fear, silence, love,
   time) are never targets.
2. The situation is opened by the wording of the prompt (a negation, a
   mental-state verb, a comparison, a barrier or viewpoint). The prompt itself, not
   world knowledge, places the target outside the scene.
3. The prompt does not elsewhere ask for the same target to appear normally in the
   scene. If it does, the prompt is negative for that target.
4. The prompt describes a scene. Pure style or keyword lists with no scene
   ("8k, trending on artstation, by greg rutkowski") are negative.

## Family 1: existence-canceling (`existence`)

The target is mentioned under an operator that cancels its existence in the
described scene: *no*, *not*, *without*, *never*, *-less* used on an entity, *empty
of*, *free of*.

- Positive: "a beach at sunset with no people" (target: people); "a kitchen
  without a refrigerator" (refrigerator); "a treeless hill" (trees).
- Negative, and mark `meta_negation` instead: negations of image-quality or
  rendering artefacts rather than scene content: *no watermark, no text, no
  signature, no blur, no deformed hands, no extra fingers, no frame, no border,
  not cropped*. These are instructions to the renderer, not scene content.
- Negative: abstract targets ("without fear"), idioms ("not only ... but"),
  "no" as an answer or as part of a title.

## Family 2: mental-state (`mental`)

The target is mentioned inside the mental state of a person **in the scene**: what
that person believes, thinks, remembers, imagines, dreams, fears, hopes, wishes,
pretends, or hallucinates. The target exists only in that person's mind.

- Positive: "a little girl dreaming of a dragon" (dragon); "an old man
  remembering his wedding day" (wedding); "a boy who believes a monster is under
  his bed" (monster); "a cat imagining itself as a lion" (lion).
- Negative: the mental-state verb addresses the *generator or viewer*, not a
  person in the scene: "imagine a city floating in the sky", "dream of a world
  where...", "picture a forest". These request the scene itself.
- Negative: adjectives of mood or style: *dreamy, dreamlike, imaginative,
  surreal, nostalgic, hopeful*.
- Negative: the target is also physically present ("a girl looking at a dragon
  and thinking about it").

## Family 3: figurative (`figurative`)

The target is the **vehicle** of a comparison: it characterizes something else in
the scene and does not belong to the scene itself. Cues: *like a*, *as ... as*,
*resembles*, *as if*, *-like*, *-shaped* when the shape is borrowed from an entity,
*reminiscent of an X*.

- Positive: "a man with a face like a bulldog" (bulldog); "hair as red as fire"
  (fire); "a fortress resembling a sleeping dragon" (dragon); "cat-like eyes"
  (cat); "a building shaped like a giant shoe" (shoe).
- Negative: comparisons to a **style, artist, medium, or work**: "like a Pixar
  movie", "in the style of Van Gogh", "looks like an oil painting", "like a
  1980s photograph", "like a Wes Anderson film". The vehicle is a style, not an
  entity.
- Negative: "like" for examples, lists, preference, or approximation ("things
  like cars and boats", "something like a castle", "I like").
- Negative: "as" without a comparison ("dressed as a pirate" is a costume: the
  pirate outfit is requested, so it is negative; "as a child" is time).
- Note: the paper allows a stylized rendering to show the vehicle. That does not
  matter here; classify by the prompt only.

## Family 4: perspectival (`perspectival`)

The target is placed where the viewpoint of the image may not reach it: inside a
closed container, behind or under something that covers it, on the far side of a
wall or door, on a surface turned away from the camera, or written on a page or
screen that faces a character rather than the viewer. Cues: *hidden, inside,
closed, sealed, wrapped, covered, behind, under, underneath, in her pocket, in a
bag, facing away, back to the camera, out of view, from behind*.

- Positive: "a child hiding a puppy inside a closed box" (puppy); "a woman with a
  letter in her coat pocket" (letter); "a man seen from behind reading a
  newspaper" (the newspaper's content); "a present wrapped in gold paper, a watch
  inside" (watch); "a sign facing away from the camera that reads OPEN" (the word
  OPEN).
- Negative: *inside* or *behind* when the target is still plainly in view:
  "inside a cathedral" (the interior is the scene), "mountains behind the house",
  "a cat under a table" (visible under an open table). Ask: does the wording
  imply the target is out of sight from the camera? Only then positive.
- Negative: *hidden* as a figure of speech ("hidden gem", "hidden meaning",
  "hidden valley" as a place name).
- Text targets: positive only when the prompt states that the writing faces a
  character or away from the viewer, or is inside something closed. "a sign that
  says HELLO" with no such wording is negative (the text is requested).

## Output

One JSON object per prompt, on its own line, in the order given:

```
{"wid": "...", "family": "existence" | "mental" | "figurative" | "perspectival" | "none",
 "target": "...", "cue": "...", "confidence": "high" | "medium" | "low",
 "flag": "" | "meta_negation" | "style_comparison" | "generator_addressed" | "target_also_requested",
 "note": "one short clause, only when not obvious"}
```

- `family` is the single family that fits best; if a prompt genuinely has two
  independent situations, output two lines with the same `wid`.
- For `none`, set `target` and `cue` to "" and use `flag` to record why a
  near-miss was rejected (this is used to estimate the false-positive types).
- `target` is the noun phrase as it appears in the prompt, minimally trimmed.
- `cue` is the exact word or phrase that opens the situation.
- Be strict. A prompt that needs an argument to count is `low` confidence or
  `none`. We would rather miss a case than admit a doubtful one.
