# Over-realization in the wild (Opus judge, protocol final)

Wild items: Existence-canceling 62 random + 0 targeted, Mental-state 3 random + 135 targeted, Figurative 129 random + 0 targeted, Perspectival 8 random + 60 targeted

Rates over judged images (label != other). OR = DO + SO. Gen = OverReal-Gen eval sample, same judge.

## Existence-canceling

| system | wild n | wild OR | DO | SO | I | W | Gen n | Gen OR | Gen DO | Gen SO |
|---|---|---|---|---|---|---|---|---|---|---|
| GPT-Image | 60 | 0.28 | 0.28 | 0.00 | 0.00 | 0.72 | 100 | 0.02 | 0.02 | 0.00 |
| Nano Banana | 60 | 0.38 | 0.38 | 0.00 | 0.00 | 0.62 | 50 | 0.12 | 0.12 | 0.00 |
| Ideogram | 62 | 0.55 | 0.55 | 0.00 | 0.00 | 0.45 | 50 | 0.08 | 0.08 | 0.00 |
| FLUX.1-dev | 122 | 0.70 | 0.70 | 0.00 | 0.00 | 0.30 | 100 | 0.43 | 0.43 | 0.00 |
| Qwen-Image | 61 | 0.72 | 0.72 | 0.00 | 0.00 | 0.28 | 100 | 0.64 | 0.64 | 0.00 |
| OmniGen2 | 61 | 0.67 | 0.67 | 0.00 | 0.00 | 0.33 | 100 | 0.51 | 0.51 | 0.00 |
| SD3.5-Medium | 62 | 0.69 | 0.69 | 0.00 | 0.00 | 0.31 | 100 | 0.34 | 0.34 | 0.00 |
| SD3.5-Large | 61 | 0.70 | 0.70 | 0.00 | 0.00 | 0.30 | 100 | 0.64 | 0.64 | 0.00 |

Pooled over systems: random-sample items OR = 0.60 (n=549); all wild items OR = 0.60 (n=549).

## Mental-state

| system | wild n | wild OR | DO | SO | I | W | Gen n | Gen OR | Gen DO | Gen SO |
|---|---|---|---|---|---|---|---|---|---|---|
| GPT-Image | 127 | 0.33 | 0.21 | 0.12 | 0.64 | 0.03 | 100 | 0.61 | 0.35 | 0.26 |
| Nano Banana | 135 | 0.36 | 0.26 | 0.10 | 0.58 | 0.06 | 75 | 0.56 | 0.37 | 0.19 |
| Ideogram | 138 | 0.44 | 0.28 | 0.16 | 0.14 | 0.41 | 75 | 0.53 | 0.29 | 0.24 |
| FLUX.1-dev | 276 | 0.47 | 0.30 | 0.17 | 0.04 | 0.49 | 99 | 0.73 | 0.44 | 0.28 |
| Qwen-Image | 138 | 0.75 | 0.51 | 0.24 | 0.06 | 0.19 | 99 | 0.89 | 0.67 | 0.22 |
| OmniGen2 | 137 | 0.60 | 0.42 | 0.18 | 0.04 | 0.36 | 100 | 0.86 | 0.64 | 0.22 |
| SD3.5-Medium | 138 | 0.53 | 0.33 | 0.20 | 0.03 | 0.44 | 100 | 0.87 | 0.66 | 0.21 |
| SD3.5-Large | 138 | 0.59 | 0.38 | 0.21 | 0.03 | 0.38 | 100 | 0.84 | 0.61 | 0.23 |

Pooled over systems: random-sample items OR = 0.37 (n=27); all wild items OR = 0.51 (n=1227).

## Figurative

| system | wild n | wild OR | DO | SO | I | W | Gen n | Gen OR | Gen DO | Gen SO |
|---|---|---|---|---|---|---|---|---|---|---|
| GPT-Image | 119 | 0.22 | 0.01 | 0.21 | 0.47 | 0.31 | 100 | 0.51 | 0.22 | 0.29 |
| Nano Banana | 123 | 0.25 | 0.01 | 0.24 | 0.35 | 0.40 | 99 | 0.41 | 0.15 | 0.26 |
| Ideogram | 128 | 0.29 | 0.02 | 0.27 | 0.26 | 0.45 | 100 | 0.32 | 0.07 | 0.25 |
| FLUX.1-dev | 136 | 0.32 | 0.02 | 0.29 | 0.14 | 0.54 | 99 | 0.39 | 0.09 | 0.30 |
| Qwen-Image | 129 | 0.30 | 0.04 | 0.26 | 0.36 | 0.34 | 99 | 0.62 | 0.28 | 0.33 |
| OmniGen2 | 129 | 0.28 | 0.02 | 0.26 | 0.26 | 0.46 | 97 | 0.45 | 0.19 | 0.27 |
| SD3.5-Medium | 129 | 0.29 | 0.03 | 0.26 | 0.23 | 0.47 | 99 | 0.40 | 0.09 | 0.31 |
| SD3.5-Large | 129 | 0.28 | 0.02 | 0.26 | 0.29 | 0.43 | 99 | 0.61 | 0.20 | 0.40 |

Pooled over systems: random-sample items OR = 0.28 (n=1022); all wild items OR = 0.28 (n=1022).

## Perspectival

| system | wild n | wild OR | DO | SO | I | W | Gen n | Gen OR | Gen DO | Gen SO |
|---|---|---|---|---|---|---|---|---|---|---|
| GPT-Image | 66 | 0.82 | 0.59 | 0.23 | 0.00 | 0.18 | 99 | 0.69 | 0.54 | 0.15 |
| Nano Banana | 67 | 0.78 | 0.70 | 0.07 | 0.01 | 0.21 | 100 | 0.95 | 0.78 | 0.17 |
| Ideogram | 68 | 0.88 | 0.74 | 0.15 | 0.00 | 0.12 | 100 | 0.83 | 0.65 | 0.18 |
| FLUX.1-dev | 68 | 0.78 | 0.65 | 0.13 | 0.00 | 0.22 | 100 | 0.75 | 0.56 | 0.19 |
| Qwen-Image | 67 | 0.87 | 0.75 | 0.12 | 0.00 | 0.13 | 100 | 0.89 | 0.67 | 0.22 |
| OmniGen2 | 67 | 0.75 | 0.66 | 0.09 | 0.00 | 0.25 | 98 | 0.81 | 0.64 | 0.16 |
| SD3.5-Medium | 68 | 0.74 | 0.63 | 0.10 | 0.00 | 0.26 | 99 | 0.78 | 0.67 | 0.11 |
| SD3.5-Large | 68 | 0.79 | 0.69 | 0.10 | 0.01 | 0.19 | 98 | 0.76 | 0.59 | 0.16 |

Pooled over systems: random-sample items OR = 0.90 (n=63); all wild items OR = 0.80 (n=539).

