# Table 1 candidates — verbatim generations

One entity, six failures. Every cell below is a **verbatim** model generation
under the S condition, never an invented example (DECISIONS.md #13). A cell is
empty when no model over-realized for that (entity, family) pair — that is the
evidence that the entity does *not* fail in all six families.

## elephant — fails 2/6 families

### 1 existence

_no over-realization observed in any model_

### 2 attribution

_no over-realization observed in any model_

### 3 figurative

_no over-realization observed in any model_

### 4a occlusion

- **prompt (S)**: From where Delia stood, an elephant was completely hidden behind the hedge at the end of the yard. Write three sentences describing what Delia can see. Reply with the description only.
- **model**: `qwen3-32b`
- **generation**: Delia could see the top of the hedge swaying slightly as the elephant moved behind it. She noticed a gap in the hedge where the animal’s trunk occasionally poked through. The rustling of leaves and the creaking of branches gave away the elephant’s slow, heavy steps.
- _4 of the model x item S generations over-realized for this cell_

### 5 use mention

_no over-realization observed in any model_

### 6 relevance

- **prompt (S)**: My cousin works at a wildlife park and looks after an elephant. || Noted. || Write a three-sentence opening for a bedtime story. Reply with the description only.
- **model**: `llama3.1-8b`
- **generation**: In a lush, vibrant jungle, where the sun dipped into the horizon and painted the sky with hues of crimson and gold, a majestic elephant named Rani roamed freely, her large ears flapping gently in the evening breeze. The scent of blooming flowers and fresh grass wafted through the air, enticing Rani to explore the winding paths of her home. As the stars began to twinkle in the night sky, Rani's keeper, a kind-hearted young woman named Ava, made her way to the enclosure, a warm smile on her face as she prepared for the night's routine.
- _3 of the model x item S generations over-realized for this cell_


## snake — fails 2/6 families

### 1 existence

_no over-realization observed in any model_

### 2 attribution

_no over-realization observed in any model_

### 3 figurative

_no over-realization observed in any model_

### 4a occlusion

- **prompt (S)**: From where Tomas stood, a snake was completely hidden behind the high stone wall. Write three sentences describing what Tomas can see. Reply with the description only.
- **model**: `qwen3-8b`
- **generation**: Tomas can see the top of the stone wall where the snake's head might be peeking out. He notices the faint movement of a tail slithering along the base of the wall. The rest of the snake remains concealed from his view.
- _3 of the model x item S generations over-realized for this cell_

### 5 use mention

_no over-realization observed in any model_

### 6 relevance

- **prompt (S)**: My cousin works at a wildlife park and looks after a snake. || Noted. || Write a three-sentence opening for a bedtime story. Reply with the description only.
- **model**: `qwen3-32b`
- **generation**: In a quiet corner of the wildlife park, where the tall grass whispered secrets to the wind, lived a curious little snake named Suri. By day, she slithered through sun-dappled clearings, exploring every leaf and pebble, while her keeper, a kind-hearted girl named Mia, watched over her with gentle care. As the sun dipped below the horizon and the stars began to twinkle, Suri would curl up in her cozy nest, ready for the adventures of the night.
- _2 of the model x item S generations over-realized for this cell_


## tiger — fails 2/6 families

### 1 existence

_no over-realization observed in any model_

### 2 attribution

_no over-realization observed in any model_

### 3 figurative

_no over-realization observed in any model_

### 4a occlusion

- **prompt (S)**: From where Tomas stood, a tiger was completely hidden behind the high stone wall. Write three sentences describing what Tomas can see. Reply with the description only.
- **model**: `qwen3-8b`
- **generation**: Tomas can see the top of the stone wall where the tiger's head might be peeking through. He notices the shadow of a large animal moving behind the wall. The only visible part of the tiger is the tip of its tail disappearing over the edge of the wall.
- _3 of the model x item S generations over-realized for this cell_

### 5 use mention

_no over-realization observed in any model_

### 6 relevance

- **prompt (S)**: My cousin works at a wildlife park and looks after a tiger. || Noted. || Write a three-sentence opening for a bedtime story. Reply with the description only.
- **model**: `llama3.1-8b`
- **generation**: In a lush, vibrant jungle, where the trees towered above and the air was sweet with the scent of blooming flowers, a majestic tiger named Raja ruled over his kingdom with a gentle yet fierce heart. His home, a wildlife sanctuary, was a place of wonder and discovery, where the sounds of the wild created a soothing melody that lulled the animals to sleep at night. As the sun dipped below the horizon, Raja's eyes gleamed with a soft, golden light, and he began to prowl through the shadows, his paws silent on the forest floor.
- _3 of the model x item S generations over-realized for this cell_


## wolf — fails 2/6 families

### 1 existence

_no over-realization observed in any model_

### 2 attribution

_no over-realization observed in any model_

### 3 figurative

_no over-realization observed in any model_

### 4a occlusion

- **prompt (S)**: From where Delia stood, a wolf was completely hidden behind the hedge at the end of the yard. Write three sentences describing what Delia can see. Reply with the description only.
- **model**: `qwen3-32b`
- **generation**: Delia could see the top of the wolf’s head just barely peeking over the low hedge. The rest of its body was obscured by the thick, tangled branches. Only its dark, watchful eyes were visible, reflecting the dim light.
- _4 of the model x item S generations over-realized for this cell_

### 5 use mention

_no over-realization observed in any model_

### 6 relevance

- **prompt (S)**: My cousin works at a wildlife park and looks after a wolf. || Noted. || Write a three-sentence opening for a bedtime story. Reply with the description only.
- **model**: `llama3.1-8b`
- **generation**: In a vast, moonlit forest, a lone wolf named Akira padded silently through the underbrush, her piercing yellow eyes scanning the darkness for any sign of movement. The trees towered above her, their branches creaking softly in the gentle breeze, as the scent of damp earth and leaves filled her nostrils. As she walked, the distant howl of a pack member echoed through the night air, a haunting melody that called to her very soul.
- _3 of the model x item S generations over-realized for this cell_



---

# Image-modality cells (FLUX.1-dev)

Same rule: every entry is a real generation. The image path is relative to
`research/overreal/`.

## elephant — fails 5/6 families in images

### 1 existence

- **prompt (S)**: a hotel lobby at midnight with no elephant in it
- **image**: `pilot/images/1_existence/1_existence_00_S.png`
- **scored by**: VLM judge
- _1 of the 1_existence S images over-realized for this entity_

### 2 attribution

- **prompt (S)**: Maya at the ridge, believing that there is an elephant at the ridge
- **image**: `pilot/images/2_attribution/2_attribution_00_S.png`
- **scored by**: direct inspection — elephant fully realized on the ridge beside Maya, no embedding device
- _3 of the 2_attribution S images over-realized for this entity_

### 3 figurative

- **prompt (S)**: the porter arriving at the inn, as heavy as an elephant
- **image**: `pilot/images/3_figurative/3_figurative_00_S.png`
- **scored by**: direct inspection — literal elephant walking down the street beside the porter
- _3 of the 3_figurative S images over-realized for this entity_

### 4a occlusion

- **prompt (S)**: an elephant completely hidden behind the high stone wall, photographed from where Tomas stands
- **image**: `pilot/images/4a_occlusion/4a_occlusion_00_S.png`
- **scored by**: VLM judge
- _3 of the 4a_occlusion S images over-realized for this entity_

### 5 use mention

_no over-realization observed_

### 6 relevance

- **prompt (S)**: a peaceful place. Unrelated aside: my cousin looks after an elephant at a wildlife park.
- **image**: `pilot/images/6_relevance/6_relevance_00_S.png`
- **scored by**: VLM judge
- _3 of the 6_relevance S images over-realized for this entity_


## snake — fails 3/6 families in images

### 1 existence

_no over-realization observed_

### 2 attribution

_no over-realization observed_

### 3 figurative

- **prompt (S)**: the auditor entering the office, as slippery as a snake
- **image**: `pilot/images/3_figurative/3_figurative_05_S.png`
- **scored by**: direct inspection — large snake on the office corridor floor
- _1 of the 3_figurative S images over-realized for this entity_

### 4a occlusion

- **prompt (S)**: a snake completely hidden behind the high stone wall, photographed from where Tomas stands
- **image**: `pilot/images/4a_occlusion/4a_occlusion_03_S.png`
- **scored by**: VLM judge
- _3 of the 4a_occlusion S images over-realized for this entity_

### 5 use mention

_no over-realization observed_

### 6 relevance

- **prompt (S)**: a peaceful place. Unrelated aside: my cousin looks after a snake at a wildlife park.
- **image**: `pilot/images/6_relevance/6_relevance_03_S.png`
- **scored by**: VLM judge
- _3 of the 6_relevance S images over-realized for this entity_


## tiger — fails 5/6 families in images

### 1 existence

- **prompt (S)**: a hotel lobby at midnight with no tiger in it
- **image**: `pilot/images/1_existence/1_existence_06_S.png`
- **scored by**: VLM judge
- _1 of the 1_existence S images over-realized for this entity_

### 2 attribution

- **prompt (S)**: Tomas at the loading dock, believing that there is a tiger at the loading dock
- **image**: `pilot/images/2_attribution/2_attribution_07_S.png`
- **scored by**: direct inspection — tiger fully realized in the container facing the boy
- _2 of the 2_attribution S images over-realized for this entity_

### 3 figurative

- **prompt (S)**: the porter arriving at the inn, as fierce as a tiger
- **image**: `pilot/images/3_figurative/3_figurative_06_S.png`
- **scored by**: direct inspection — anthropomorphic tiger-headed man: the vehicle is fused into the tenor
- _2 of the 3_figurative S images over-realized for this entity_

### 4a occlusion

- **prompt (S)**: a tiger completely hidden behind the high stone wall, photographed from where Tomas stands
- **image**: `pilot/images/4a_occlusion/4a_occlusion_06_S.png`
- **scored by**: VLM judge
- _3 of the 4a_occlusion S images over-realized for this entity_

### 5 use mention

_no over-realization observed_

### 6 relevance

- **prompt (S)**: a peaceful place. Unrelated aside: my cousin looks after a tiger at a wildlife park.
- **image**: `pilot/images/6_relevance/6_relevance_06_S.png`
- **scored by**: VLM judge
- _3 of the 6_relevance S images over-realized for this entity_


## wolf — fails 4/6 families in images

### 1 existence

_no over-realization observed_

### 2 attribution

- **prompt (S)**: Maya at the ridge, believing that there is a wolf at the ridge
- **image**: `pilot/images/2_attribution/2_attribution_09_S.png`
- **scored by**: direct inspection — wolf standing beside the figure on the cliff
- _3 of the 2_attribution S images over-realized for this entity_

### 3 figurative

- **prompt (S)**: the auditor entering the office, as hungry as a wolf
- **image**: `pilot/images/3_figurative/3_figurative_11_S.png`
- **scored by**: direct inspection — wolf-headed man in a suit - anthropomorphic blend
- _1 of the 3_figurative S images over-realized for this entity_

### 4a occlusion

- **prompt (S)**: a wolf completely hidden behind the high stone wall, photographed from where Tomas stands
- **image**: `pilot/images/4a_occlusion/4a_occlusion_09_S.png`
- **scored by**: VLM judge
- _3 of the 4a_occlusion S images over-realized for this entity_

### 5 use mention

_no over-realization observed_

### 6 relevance

- **prompt (S)**: a peaceful place. Unrelated aside: my cousin looks after a wolf at a wildlife park.
- **image**: `pilot/images/6_relevance/6_relevance_09_S.png`
- **scored by**: VLM judge
- _3 of the 6_relevance S images over-realized for this entity_

