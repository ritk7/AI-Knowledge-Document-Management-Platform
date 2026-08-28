Evaluated on 30 answerable questions (5 additional unanswerable questions used for threshold calibration) over a 3-document corpus.

| Configuration | P@1 | R@1 | P@3 | R@3 | P@5 | R@5 | hit@3 | MRR |
|---------------|----:|----:|----:|----:|----:|----:|------:|----:|
| Vector only | 0.767 | 0.733 | 0.322 | 0.900 | 0.207 | 0.950 | 0.933 | 0.859 |
| BM25 only | 0.700 | 0.633 | 0.311 | 0.817 | 0.193 | 0.850 | 0.833 | 0.775 |
| Hybrid (a=0.6) | 0.833 | 0.767 | 0.367 | 0.983 | 0.220 | 0.983 | 1.000 | 0.906 |
| Hybrid + rerank | 0.933 | 0.867 | 0.356 | 0.950 | 0.213 | 0.950 | 0.967 | 0.950 |

**hit@3 by question type** — where each retriever earns its place:

| Question type | n | Vector only | BM25 only | Hybrid + rerank |
|---------------|--:|------------:|----------:|----------------:|
| lexical | 8 | 0.750 | 1.000 | 1.000 |
| semantic | 10 | 1.000 | 0.600 | 0.900 |
| mixed | 12 | 1.000 | 0.917 | 1.000 |

### Abstention calibration

Neither confidence signal separates answerable from unanswerable on its own:

| | n | cross-encoder mean | median | p10 | cosine mean | cosine max |
|---|--:|-------------------:|-------:|----:|------------:|-----------:|
| Answerable | 30 | 0.626 | 0.977 | 0.000 | 0.461 | — |
| Unanswerable | 5 | 0.030 | — | — | 0.294 | 0.540 |

The cross-encoder has a **median of 0.977 but a p10 of 0.000** on answerable questions — bimodal, because it scores correct-but-paraphrased passages near zero. Cosine survives paraphrase but rates topically-adjacent irrelevant chunks highly (max 0.540 on unanswerable, above many answerable ones).

Because they fail on *different* questions, the shipped gate abstains only when **both** are weak. `rerank-only` rows below are the single-signal baseline:

| t_rerank | t_cosine | Catches unanswerable | Wrongly refuses answerable |
|---------:|---------:|---------------------:|---------------------------:|
| 0.02 | 0.30 ✅ | 3/5 (60%) | 4/30 (13%) |
| 0.02 | rerank only | 3/5 (60%) | 8/30 (27%) |
| 0.05 | rerank only | 4/5 (80%) | 9/30 (30%) |
| 0.10 | rerank only | 4/5 (80%) | 9/30 (30%) |
| 0.15 | rerank only | 5/5 (100%) | 9/30 (30%) |
| 0.20 | rerank only | 5/5 (100%) | 9/30 (30%) |
| 0.30 | rerank only | 5/5 (100%) | 11/30 (37%) |

**Shipped defaults: `CONFIDENCE_THRESHOLD=0.15`, `VECTOR_CONFIDENCE_THRESHOLD=0.3`** — catches 3/5 unanswerable questions while wrongly refusing 4/30 answerable ones.

The gate is a cheap pre-filter, not the only guardrail: questions that pass it still reach a model instructed to reply "I don't have enough information" when the excerpts don't cover the question. The threshold exists to avoid paying for that call when retrieval clearly found nothing.

> Calibrated on only 5 unanswerable questions — enough to show the single-signal gate is unusable, not enough to fix these thresholds precisely. Re-run against your own corpus before relying on them.
