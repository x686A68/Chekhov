# Q1 cross-modal — does any entity fail in all six families?

Each cell is `text S / image S`: the suppression-failure rate under the S
condition in each modality. Text pools the three models; images are FLUX.1-dev,
scored by direct inspection for families 2 and 3 and by the VLM judge otherwise.
**bold** marks a cell where the entity failed in at least one modality.

| entity | 1 existence | 2 attribution | 3 figurative | 4a occlusion | 5 use mention | 6 relevance | families failed |
|---|---|---|---|---|---|---|---|
| **elephant** | **0.00 / 0.33** | **0.00 / 1.00** | **0.00 / 1.00** | **0.44 / 1.00** | 0.00 / 0.00 | **0.33 / 1.00** | **5/6** |
| **snake** | 0.00 / 0.00 | 0.00 / 0.00 | **0.00 / 0.33** | **0.33 / 1.00** | 0.00 / 0.00 | **0.22 / 1.00** | **3/6** |
| **tiger** | **0.00 / 0.33** | **0.00 / 0.67** | **0.00 / 0.67** | **0.33 / 1.00** | 0.00 / 0.00 | **0.33 / 1.00** | **5/6** |
| **wolf** | 0.00 / 0.00 | **0.00 / 1.00** | **0.00 / 0.33** | **0.44 / 1.00** | 0.00 / 0.00 | **0.33 / 1.00** | **4/6** |

## By family, pooled over entities

| family | text S | image S | text Δ | image Δ |
|---|---|---|---|---|
| 1 existence | 0.00 | 0.17 | +0.97 | +0.75 |
| 2 attribution | 0.00 | 0.67 | +0.53 | +0.17 |
| 3 figurative | 0.00 | 0.58 | +0.72 | +0.42 |
| 4a occlusion | 0.39 | 1.00 | +0.61 | -0.17 |
| 5 use mention | 0.00 | 0.00 | +0.94 | +1.00 |
| 6 relevance | 0.31 | 1.00 | +0.67 | -0.08 |
