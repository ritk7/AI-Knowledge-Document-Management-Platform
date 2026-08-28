from pydantic import BaseModel, Field


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    num_chunks: int
    num_pages: int = 1


class UploadResponse(BaseModel):
    document: DocumentInfo
    message: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]


class QueryRequest(BaseModel):
    question: str = Field(max_length=4000)
    # ge=1 matters: top_k=0 previously fell through Python's `0 or default`
    # and silently returned the *default* count instead of zero or an error.
    # Bounding it here means the bad value never reaches retrieval logic.
    top_k: int | None = Field(default=None, ge=1, le=50)
    document_id: str | None = None
    # Hybrid weighting override from the UI slider. 1.0 = pure vector,
    # 0.0 = pure BM25. Falls back to the server default when omitted.
    alpha: float | None = Field(default=None, ge=0.0, le=1.0)
    use_reranker: bool = True


class SourceChunk(BaseModel):
    """One retrieved chunk, with the score from every pipeline stage.

    Scores are surfaced so the UI can show *why* a chunk was retrieved --
    whether it won on semantic similarity, keyword match, or both.
    """

    citation: int  # 1-indexed marker the LLM cites as [1], [2], ...
    document_id: str
    filename: str
    page: int
    chunk_index: int
    text: str

    vector_score: float | None = None  # cosine similarity, 0-1
    bm25_score: float | None = None  # raw Okapi BM25 score, unbounded
    fused_score: float | None = None  # normalized weighted combination
    rerank_score: float | None = None  # cross-encoder relevance, 0-1


class RetrievalStats(BaseModel):
    alpha: float
    reranked: bool
    confidence: float  # cross-encoder score on the top chunk
    semantic_confidence: float  # dense cosine similarity on the best chunk
    confidence_threshold: float
    vector_confidence_threshold: float
    candidates_vector: int
    candidates_bm25: int
    candidates_fused: int
    latency_ms: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    abstained: bool = False
    stats: RetrievalStats | None = None
