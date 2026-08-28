"""Cross-encoder re-ranking stage.

Bi-encoder vs cross-encoder
---------------------------
The retrieval embeddings (all-MiniLM-L6-v2) are a *bi-encoder*: query and
passage are encoded separately and compared by cosine distance. That
independence is what makes the index possible -- passage vectors are computed
once at upload time -- but it also means the model never sees the query and the
passage together, so it cannot reason about how they relate.

A *cross-encoder* concatenates them into a single sequence, `[CLS] query [SEP]
passage [SEP]`, and runs full self-attention across both. Every passage token
can attend to every query token, which resolves exactly the cases the bi-encoder
gets wrong -- negation, pronoun binding, and passages that are topically
on-point but do not actually answer the question.

The cost is that relevance can no longer be precomputed: scoring N candidates
takes N forward passes. So the cross-encoder never touches the corpus. It only
re-orders the handful of candidates that survive hybrid fusion, which is where
its accuracy buys the most per unit of compute.

The model emits an unbounded relevance logit; a sigmoid maps it to [0, 1], which
we use both to rank and as the confidence signal for abstention.
"""

from __future__ import annotations

import math
from functools import lru_cache

from sentence_transformers import CrossEncoder

from app.config import settings


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    """Loaded lazily -- the model is only downloaded when reranking first runs."""
    return CrossEncoder(settings.reranker_model)


def _sigmoid(x: float) -> float:
    # Guard against overflow on large-magnitude logits.
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def rerank(
    question: str, candidates: list[dict], top_k: int
) -> list[tuple[dict, float]]:
    """Reorder candidates by cross-encoder relevance.

    Args:
        question: the user's query.
        candidates: chunk dicts, each with a "text" key.
        top_k: how many to keep.

    Returns:
        (chunk, confidence) pairs sorted best-first, where confidence is the
        sigmoid of the cross-encoder logit in [0, 1].
    """
    if not candidates:
        return []

    pairs = [(question, chunk["text"]) for chunk in candidates]
    logits = get_reranker().predict(pairs)

    scored = [
        (chunk, _sigmoid(float(logit))) for chunk, logit in zip(candidates, logits)
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_k]
