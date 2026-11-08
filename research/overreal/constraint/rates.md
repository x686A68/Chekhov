# Explicit-constraint prompts: outcome shares (other kept in the denominator)

Constraint = original prompt + "No <target> should appear in the image.", seed 0.
Baseline = raw prompts of the same items in the Gen eval sample (both seeds).

## Mental-state

| Model | Prompts | n | target present | DO | SO | I | target absent | W | other |
|---|---|---|---|---|---|---|---|---|---|
| FLUX.1-dev | raw | 100 | 0.75 | 0.44 | 0.28 | 0.03 | 0.25 | 0.24 | 0.01 |
| FLUX.1-dev | constraint | 75 | 0.76 | 0.49 | 0.23 | 0.04 | 0.24 | 0.23 | 0.01 |
| Qwen-Image | raw | 100 | 0.89 | 0.66 | 0.22 | 0.01 | 0.11 | 0.10 | 0.01 |
| Qwen-Image | constraint | 75 | 0.91 | 0.72 | 0.19 | 0.00 | 0.09 | 0.09 | 0.00 |
| OmniGen2 | raw | 100 | 0.89 | 0.64 | 0.22 | 0.03 | 0.11 | 0.11 | 0.00 |
| OmniGen2 | constraint | 75 | 0.83 | 0.60 | 0.20 | 0.03 | 0.17 | 0.17 | 0.00 |
| SD3.5-Medium | raw | 100 | 0.88 | 0.66 | 0.21 | 0.01 | 0.12 | 0.12 | 0.00 |
| SD3.5-Medium | constraint | 75 | 0.87 | 0.44 | 0.43 | 0.00 | 0.13 | 0.13 | 0.00 |
| SD3.5-Large | raw | 100 | 0.87 | 0.61 | 0.23 | 0.03 | 0.13 | 0.13 | 0.00 |
| SD3.5-Large | constraint | 75 | 0.87 | 0.68 | 0.19 | 0.00 | 0.13 | 0.13 | 0.00 |
| Pooled | raw | 500 | 0.86 | 0.60 | 0.23 | 0.02 | 0.14 | 0.14 | 0.00 |
| Pooled | constraint | 375 | 0.85 | 0.59 | 0.25 | 0.01 | 0.15 | 0.15 | 0.00 |

## Figurative

| Model | Prompts | n | target present | DO | SO | I | target absent | W | other |
|---|---|---|---|---|---|---|---|---|---|
| FLUX.1-dev | raw | 47 | 0.47 | 0.09 | 0.23 | 0.15 | 0.53 | 0.53 | 0.00 |
| FLUX.1-dev | constraint | 118 | 0.58 | 0.14 | 0.36 | 0.07 | 0.42 | 0.41 | 0.02 |
| Qwen-Image | raw | 45 | 0.69 | 0.27 | 0.38 | 0.04 | 0.31 | 0.31 | 0.00 |
| Qwen-Image | constraint | 118 | 0.76 | 0.26 | 0.43 | 0.07 | 0.24 | 0.24 | 0.00 |
| OmniGen2 | raw | 52 | 0.50 | 0.12 | 0.31 | 0.08 | 0.50 | 0.48 | 0.02 |
| OmniGen2 | constraint | 118 | 0.58 | 0.21 | 0.31 | 0.07 | 0.42 | 0.38 | 0.03 |
| SD3.5-Medium | raw | 61 | 0.39 | 0.08 | 0.25 | 0.07 | 0.61 | 0.59 | 0.02 |
| SD3.5-Medium | constraint | 118 | 0.58 | 0.17 | 0.35 | 0.06 | 0.42 | 0.41 | 0.02 |
| SD3.5-Large | raw | 54 | 0.67 | 0.17 | 0.37 | 0.13 | 0.33 | 0.31 | 0.02 |
| SD3.5-Large | constraint | 118 | 0.74 | 0.25 | 0.39 | 0.10 | 0.26 | 0.25 | 0.01 |
| Pooled | raw | 259 | 0.54 | 0.14 | 0.31 | 0.09 | 0.46 | 0.45 | 0.01 |
| Pooled | constraint | 590 | 0.65 | 0.21 | 0.37 | 0.07 | 0.35 | 0.34 | 0.02 |
