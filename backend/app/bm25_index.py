"""BM25 keyword index, the lexical half of hybrid retrieval.

Why this exists
---------------
Dense embeddings are strong on paraphrase and weak on *rare literal tokens*.
Consider a support handbook containing "Error code E-4021 indicates a TLS
handshake failure." and a user asking "what does error E-4021 mean?".

MiniLM maps `E-4021` to a near-generic subword vector -- it has no learned
representation distinguishing it from `E-4022`. Every "error code" chunk in the
corpus lands at nearly the same cosine distance, so the right chunk is retrieved
only by luck. BM25 sees `E-4021` as a term with document frequency 1, assigns it
a very high IDF, and ranks the correct chunk first with near-certainty.

The failure runs the other way too: a user asking "how do I cancel my plan?"
against a document that says "termination of service" shares no terms at all,
and BM25 scores it zero while embeddings match it easily.

Neither retriever dominates, so we run both and fuse. Because BM25 scores are
unbounded and cosine similarities live in [0, 1], the raw numbers are not
comparable -- we min-max normalize each retriever's scores within the candidate
set, then take a weighted sum controlled by `alpha`.
"""

from __future__ import annotations

import re
import threading

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, keeping hyphenated/underscored identifiers intact.

    Keeping `e-4021` and `api_key` as single tokens is the whole point -- a
    tokenizer that split on the hyphen would turn the rare, high-IDF term into
    two common ones and throw away the signal.
    """
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """In-memory BM25 index over all chunks, rebuilt from ChromaDB on change."""

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._chunks: list[dict] = []
        self._lock = threading.Lock()

    def build(self, chunks: list[dict]) -> None:
        """(Re)build from the full chunk list. Cheap enough to redo on upload."""
        with self._lock:
            self._chunks = chunks
            corpus = [tokenize(chunk["text"]) for chunk in chunks]
            # BM25Okapi raises on an empty corpus, so guard the cold-start case.
            self._bm25 = BM25Okapi(corpus) if corpus else None

    def search(
        self, question: str, top_k: int, document_id: str | None = None
    ) -> list[tuple[dict, float]]:
        """Return up to top_k (chunk, raw_bm25_score) pairs, best first."""
        with self._lock:
            bm25, chunks = self._bm25, self._chunks

        if bm25 is None or not chunks:
            return []

        tokens = tokenize(question)
        if not tokens:
            return []

        scores = bm25.get_scores(tokens)

        ranked = [
            (chunk, float(score))
            for chunk, score in zip(chunks, scores)
            if document_id is None or chunk["metadata"]["document_id"] == document_id
        ]
        # A zero score means no query term appears in the chunk -- it carries no
        # lexical evidence, so drop it rather than padding the candidate list.
        ranked = [pair for pair in ranked if pair[1] > 0.0]
        ranked.sort(key=lambda pair: pair[1], reverse=True)
        return ranked[:top_k]

    def __len__(self) -> int:
        return len(self._chunks)


# Module-level singleton; rebuilt by refresh_bm25_index() on startup and after
# every upload or delete.
_index = BM25Index()


def get_bm25_index() -> BM25Index:
    return _index


def refresh_bm25_index() -> int:
    """Rebuild the BM25 index from ChromaDB. Returns the chunk count."""
    from app.vector_store import get_all_chunks

    chunks = get_all_chunks()
    _index.build(chunks)
    return len(chunks)
