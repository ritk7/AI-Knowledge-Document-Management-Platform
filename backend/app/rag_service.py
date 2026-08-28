"""Answer generation: grounded prompting, inline citations, and abstention.

Two guardrails against hallucination, at different layers:

1. **Retrieval-level (hard).** If the cross-encoder's confidence in the best
   chunk is below the threshold, we never call the LLM at all. A model handed
   irrelevant context will often still produce a fluent, wrong answer, so the
   cheapest and most reliable fix is not to ask. This also saves the token cost
   of a request we would have thrown away.

2. **Prompt-level (soft).** When we do call, every excerpt is numbered and the
   model is required to attach a [n] marker to each claim. Forcing a citation
   per sentence makes ungrounded statements structurally awkward to produce,
   and any that slip through are visible to the user as uncited text.
"""

from __future__ import annotations

import re

import httpx

from app.config import settings
from app.retrieval import RetrievalResult, retrieve
from app.schemas import RetrievalStats, SourceChunk

ABSTAIN_MESSAGE = (
    "I don't have enough information in the uploaded documents to answer that "
    "confidently. Try rephrasing the question, or upload a document that covers "
    "this topic."
)

SYSTEM_PROMPT = """You answer questions strictly from numbered document excerpts supplied by the user.

Rules:
1. Use ONLY the excerpts. Never use outside knowledge, and never infer facts \
that are not stated.
2. Cite every factual claim with the excerpt number in square brackets, like \
[1] or [2][3]. A sentence that states a fact without a citation is an error.
3. If the excerpts do not contain the answer, reply exactly: "I don't have \
enough information in the provided documents to answer that." Do not guess, \
and do not pad the reply with related-but-unasked-for information.
4. If the excerpts partially answer the question, answer the part they cover \
and state plainly which part is not covered.
5. Be concise. Answer the question that was asked.
6. If an excerpt contains a table or a list of similar-looking items (error \
codes, IDs, dates), find the row that EXACTLY matches every term in the \
question before answering. Adjacent rows are not the same row -- E-4021 and \
E-4011 are different codes with different meanings even though they differ by \
one digit. Re-read the exact row you are citing before writing your answer.

Example:
<excerpts>
[1] (source: handbook.md, page 2)
Remote employees must attend the Monday 9am standup.
[2] (source: handbook.md, page 3)
All standups are recorded and posted to the team channel within one hour.
</excerpts>

Question: Do I have to attend the Monday standup if I'm remote?

Answer: Yes, remote employees are required to attend the Monday 9am \
standup [1]. If you miss it, the recording is posted to the team channel \
within an hour [2]."""


def call_ollama(prompt: str) -> str:
    """One-shot generation call to a local Ollama server.

    Ollama's /api/generate has no separate system-role field the way the
    Anthropic Messages API does, so the system prompt is prepended to the user
    prompt instead. stream=False collapses the response to one JSON object
    with the full text in "response", instead of a line-per-token stream.
    """
    response = httpx.post(
        settings.ollama_url,
        json={
            "model": settings.ollama_model,
            "prompt": f"{SYSTEM_PROMPT}\n\n{prompt}",
            "stream": False,
            # num_predict: explicit, not left to Ollama's server default
            # (undocumented and version-dependent) -- mirrors the
            # max_tokens=1024 the original Anthropic call used.
            #
            # No temperature override. Tried temperature=0.1 to reduce a
            # measured hallucination (see README) -- it made the SAME failure
            # WORSE (7/8 wrong at default -> 8/8 wrong, identical text every
            # time, at 0.1). The wrong answer was apparently the model's
            # single highest-probability completion for this input, so
            # narrowing the sampling distribution collapsed onto it more
            # reliably instead of occasionally escaping it. Reverted; a rule 6
            # prompt addition was tried instead (see SYSTEM_PROMPT).
            "options": {"num_predict": 1024},
        },
        timeout=120.0,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def _to_sources(result: RetrievalResult) -> list[SourceChunk]:
    """Convert retrieved chunks into citation-numbered response objects."""
    sources: list[SourceChunk] = []
    for position, chunk in enumerate(result.chunks, start=1):
        metadata = chunk["metadata"]
        sources.append(
            SourceChunk(
                citation=position,
                document_id=metadata["document_id"],
                filename=metadata["filename"],
                page=metadata.get("page", 1),
                chunk_index=metadata["chunk_index"],
                text=chunk["text"],
                vector_score=chunk.get("vector_score"),
                bm25_score=chunk.get("bm25_score"),
                fused_score=chunk.get("fused_score"),
                rerank_score=chunk.get("rerank_score"),
            )
        )
    return sources


def build_prompt(question: str, sources: list[SourceChunk]) -> str:
    """Number every excerpt and label it with its document and page.

    The number is the citation handle the model writes back as [n]; the filename
    and page are what the UI renders, and are included so the model can name the
    source in prose when that reads more naturally than a bare marker.
    """
    excerpts = "\n\n".join(
        f"[{s.citation}] (source: {s.filename}, page {s.page})\n{s.text}"
        for s in sources
    )
    return (
        f"<excerpts>\n{excerpts}\n</excerpts>\n\n"
        f"Question: {question}\n\n"
        "Answer using only the excerpts above, citing each claim with its "
        "bracketed excerpt number."
    )


def extract_cited(answer: str, sources: list[SourceChunk]) -> set[int]:
    """Which citation markers the model actually used.

    Lets the UI distinguish sources that were quoted from ones that were merely
    retrieved -- a chunk the model ignored is useful signal about retrieval
    quality.
    """
    valid = {s.citation for s in sources}
    return {
        int(marker)
        for marker in re.findall(r"\[(\d+)\]", answer)
        if int(marker) in valid
    }


def _stats(result: RetrievalResult) -> RetrievalStats:
    return RetrievalStats(
        alpha=result.alpha,
        reranked=result.reranked,
        confidence=result.confidence,
        semantic_confidence=result.semantic_confidence,
        confidence_threshold=settings.confidence_threshold,
        vector_confidence_threshold=settings.vector_confidence_threshold,
        candidates_vector=result.candidates_vector,
        candidates_bm25=result.candidates_bm25,
        candidates_fused=result.candidates_fused,
        latency_ms=round(result.latency_ms, 1),
    )


def generate_answer(
    question: str,
    top_k: int | None = None,
    document_id: str | None = None,
    alpha: float | None = None,
    use_reranker: bool = True,
) -> tuple[str, list[SourceChunk], bool, RetrievalStats]:
    """Retrieve, decide whether to answer, and generate.

    Returns (answer, sources, abstained, stats).
    """
    result = retrieve(
        question,
        top_k=top_k,
        document_id=document_id,
        alpha=alpha,
        use_reranker=use_reranker,
    )
    stats = _stats(result)

    if not result.chunks:
        return (
            "No documents have been indexed yet. Upload a document first.",
            [],
            True,
            stats,
        )

    sources = _to_sources(result)

    # Guardrail 1: refuse to ask the model when BOTH retrieval signals are weak.
    # Requiring both to be low is deliberate -- either one alone produces an
    # unusable false-refusal rate (see config.py and eval/eval_results.md).
    if (
        result.confidence < settings.confidence_threshold
        and result.semantic_confidence < settings.vector_confidence_threshold
    ):
        return ABSTAIN_MESSAGE, sources, True, stats

    answer = call_ollama(build_prompt(question, sources))
    return answer, sources, False, stats
