**ASSET test set** (359 sentences, source FK grade 11.65)

| Metric | Base | Fine-tuned | Human references |
|---|---:|---:|---:|
| FK grade (lower is simpler) | 8.38 | 7.41 | 8.40 |
| BERTScore F1 vs source | 0.88 | 0.90 | 0.93 |
| Embedding similarity vs source | 0.82 | 0.82 | 0.90 |
| Numbers kept | 0.89 | 0.68 | 0.79 |
| Near-copy rate | 0.00 | 0.02 | 0.05 |
| Length vs source | 1.16 | 0.81 | 0.87 |

**Hand-written technical passages** (20 sentences, source FK grade 18.51)

| Metric | Base | Fine-tuned |
|---|---:|---:|
| FK grade (lower is simpler) | 10.62 | 10.39 |
| BERTScore F1 vs source | 0.81 | 0.84 |
| Embedding similarity vs source | 0.66 | 0.71 |
| Numbers kept | 1.00 | 0.00 |
| Near-copy rate | 0.00 | 0.00 |
| Length vs source | 1.46 | 0.80 |
