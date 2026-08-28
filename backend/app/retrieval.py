"""The retrieval pipeline: hybrid fusion, re-ranking, and confidence.

    question
       |
       +-- vector search (ChromaDB, cosine)  --> N candidates
       +-- BM25 search  (rank_bm25, Okapi)   --> N candidates
       |
       v
    min-max normalize each score set, weighted sum by alpha
       |
       v
    top rerank_pool candidates --> cross-encoder --> top_k
       |
       v
    confidence = best cross-encoder score; abstain if below threshold

Every stage's score is carried through to the response so the UI can show which
retriever actually found each chunk.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from app.bm25_index import get_bm25_index
from app.config import settings
from app.embeddings import embed_query
from app.reranker import rerank
from app.vector_store import query_chunks


@dataclass
class RetrievalResult:
    chunks: list[dict] = field(default_factory=list)
    # Cross-encoder confidence in the top chunk. Excellent at *ranking*, poorly
    # calibrated as an *absolute* score -- it collapses toward zero on correct
    # but heavily paraphrased passages.
    confidence: float = 0.0
    # Dense cosine similarity of the best chunk. Survives paraphrase, but also
    # scores topically-adjacent-yet-irrelevant chunks highly. The two signals
    # fail on different inputs, which is why the abstention gate needs both.
    semantic_confidence: float = 0.0
    alpha: float = 0.0
    reranked: bool = False
    candidates_vector: int = 0
    candidates_bm25: int = 0
    candidates_fused: int = 0
    latency_ms: float = 0.0


def _min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
    """Scale scores into [0, 1] within this candidate set.

    Vector similarities are bounded in [0, 1] while BM25 scores are unbounded
    and corpus-dependent, so a raw weighted sum would be dominated by whichever
    happened to be on a larger scale. Normalizing within the candidate set makes
    alpha mean what it claims to mean.

    When every candidate ties (including a single candidate), there is no spread
    to normalize -- they all get 1.0, so the other retriever's ordering decides.
    """
    if not scores:
        return {}
    values = scores.values()
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return {key: 1.0 for key in scores}
    return {key: (value - lo) / (hi - lo) for key, value in scores.items()}


def hybrid_search(
    question: str,
    alpha: float,
    candidate_pool: int,
    document_id: str | None = None,
) -> tuple[list[dict], int, int]:
    """Run both retrievers and fuse their rankings.

    Returns (fused_candidates, n_vector_hits, n_bm25_hits).
    """
    # --- Dense retrieval ---
    vector_hits: dict[str, dict] = {}
    vector_scores: dict[str, float] = {}

    embedding = embed_query(question)
    results = query_chunks(embedding, top_k=candidate_pool, document_id=document_id)

    ids = (results.get("ids") or [[]])[0]
    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]

    for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
        vector_hits[chunk_id] = {"id": chunk_id, "text": text, "metadata": metadata}
        # Chroma returns cosine distance; convert to similarity.
        vector_scores[chunk_id] = 1.0 - float(distance)

    # --- Lexical retrieval ---
    bm25_hits: dict[str, dict] = {}
    bm25_scores: dict[str, float] = {}

    for chunk, score in get_bm25_index().search(
        question, top_k=candidate_pool, document_id=document_id
    ):
        bm25_hits[chunk["id"]] = chunk
        bm25_scores[chunk["id"]] = score

    # --- Fusion ---
    norm_vector = _min_max_normalize(vector_scores)
    norm_bm25 = _min_max_normalize(bm25_scores)

    fused: list[dict] = []
    for chunk_id in set(norm_vector) | set(norm_bm25):
        chunk = dict(vector_hits.get(chunk_id) or bm25_hits[chunk_id])
        # A chunk missing from one retriever's list contributes 0 from that
        # side -- it was outranked there, so it should not be credited.
        v = norm_vector.get(chunk_id, 0.0)
        b = norm_bm25.get(chunk_id, 0.0)
        chunk["vector_score"] = vector_scores.get(chunk_id)
        chunk["bm25_score"] = bm25_scores.get(chunk_id)
        chunk["fused_score"] = alpha * v + (1.0 - alpha) * b
        fused.append(chunk)

    fused.sort(key=lambda c: c["fused_score"], reverse=True)
    return fused, len(vector_scores), len(bm25_scores)


def retrieve(
    question: str,
    top_k: int | None = None,
    document_id: str | None = None,
    alpha: float | None = None,
    use_reranker: bool = True,
) -> RetrievalResult:
    started = time.perf_counter()

    # `or` would treat top_k=0 as falsy and silently fall back to the default
    # -- verified live: top_k=0 returned 4 sources, not 0. `is None` is the
    # correct idiom (already used correctly for alpha, one line below).
    k = settings.top_k if top_k is None else top_k
    alpha = settings.hybrid_alpha if alpha is None else alpha

    fused, n_vector, n_bm25 = hybrid_search(
        question,
        alpha=alpha,
        candidate_pool=settings.candidate_pool,
        document_id=document_id,
    )

    if not fused:
        return RetrievalResult(
            alpha=alpha,
            reranked=False,
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    if use_reranker:
        shortlist = fused[: settings.rerank_pool]
        reranked = rerank(question, shortlist, top_k=k)
        chunks = []
        for chunk, score in reranked:
            chunk = dict(chunk)
            chunk["rerank_score"] = score
            chunks.append(chunk)
        confidence = chunks[0]["rerank_score"] if chunks else 0.0
    else:
        chunks = fused[:k]
        # Without the cross-encoder the only available signal is the fused
        # score, which is normalized within the candidate set and therefore
        # says nothing about absolute relevance -- the top hit is always ~1.0
        # even when every candidate is irrelevant. We fall back to raw cosine
        # similarity, which at least is comparable across queries.
        confidence = max((c.get("vector_score") or 0.0) for c in chunks)

    semantic_confidence = max(
        (chunk.get("vector_score") or 0.0) for chunk in chunks
    )

    return RetrievalResult(
        chunks=chunks,
        confidence=confidence,
        semantic_confidence=semantic_confidence,
        alpha=alpha,
        reranked=use_reranker,
        candidates_vector=n_vector,
        candidates_bm25=n_bm25,
        candidates_fused=len(fused),
        latency_ms=(time.perf_counter() - started) * 1000,
    )
