from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BACKEND_ROOT.parent / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Local generation via Ollama -- no API key, no per-token cost, no network
    # egress. Swap in any model Ollama serves by changing OLLAMA_MODEL.
    ollama_url: str = "http://localhost:11434/api/generate"
    ollama_model: str = "llama3.2:3b"
    embedding_model: str = "all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    chunk_size: int = 1000
    chunk_overlap: int = 150

    # --- Retrieval pipeline ---
    # Candidates pulled from each retriever (vector + BM25) before fusion.
    candidate_pool: int = 20
    # Candidates kept after fusion and handed to the cross-encoder.
    rerank_pool: int = 10
    # Chunks finally sent to the LLM.
    top_k: int = 4
    # Weight on the vector score in the hybrid fusion; BM25 gets (1 - alpha).
    hybrid_alpha: float = 0.6
    # Abstention gate. The system refuses to answer only when BOTH signals are
    # weak: the cross-encoder score AND the dense cosine similarity.
    #
    # Neither works alone -- measured on the eval set, gating on the
    # cross-encoder alone falsely refuses 30% of answerable questions (it scores
    # correct-but-paraphrased passages near zero), while gating on cosine alone
    # falsely refuses 73%. They fail on different questions, so requiring both
    # to be low cuts false refusals to ~13% while still catching the clearest
    # unanswerable cases. See eval/eval_results.md.
    confidence_threshold: float = 0.15
    vector_confidence_threshold: float = 0.30

    upload_dir: str = "data/uploads"
    chroma_dir: str = "data/chroma"
    chroma_collection: str = "documents"
    # Verified live: an upload with no cap enforced took 58 minutes to embed
    # a 34MB file (42,857 chunks) and then still failed, because ChromaDB
    # rejects a write batch that large. 20MB covers any realistic document
    # for a personal knowledge base while making pathological uploads fail
    # in milliseconds instead of the better part of an hour.
    max_upload_mb: float = 20.0

    allowed_origins: str = "*"

    @property
    def upload_path(self) -> Path:
        path = BACKEND_ROOT / self.upload_dir
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def chroma_path(self) -> Path:
        path = BACKEND_ROOT / self.chroma_dir
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def cors_origins(self) -> list[str]:
        if self.allowed_origins.strip() == "*":
            return ["*"]
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]


settings = Settings()
