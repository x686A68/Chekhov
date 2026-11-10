# In-the-wild human check: judge vs. human

Sample: 202 images; annotated so far: 202 (annotators: ['jiahao'])

| Family | n | exact agreement | κ, 4 labels | κ, over-realized vs. correct |
|---|---|---|---|---|
| Existence-canceling | 34 | 0.91 | 0.82 | 0.82 |
| Mental-state | 34 | 0.82 | 0.76 | 0.76 |
| Figurative | 34 | 0.65 | 0.53 | 0.53 |
| Perspectival | 100 | 0.81 | 0.65 | 0.69 |
| All | 202 | 0.80 | 0.70 | 0.72 |

Confusion matrix (rows: judge, columns: human), all families:

| judge \ human | disruptive | silent | integrated | withheld | other |
|---|---|---|---|---|---|
| disruptive | 87 | 1 | 9 | 3 | 1 |
| silent | 11 | 12 | 7 | 1 | 2 |
| integrated | 3 | 0 | 14 | 0 | 0 |
| withheld | 0 | 1 | 1 | 49 | 0 |
| other | 0 | 0 | 0 | 0 | 0 |
