from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.bm25_index import refresh_bm25_index
from app.config import settings
from app.routers import documents, query


@asynccontextmanager
async def lifespan(app: FastAPI):
    # BM25 lives in memory, so it has to be rebuilt from ChromaDB whenever the
    # process starts -- otherwise keyword search silently returns nothing for
    # documents uploaded in a previous run.
    count = refresh_bm25_index()
    print(f"[startup] BM25 index built over {count} chunks")
    yield


app = FastAPI(
    title="AI Knowledge & Document Management Platform",
    description=(
        "Hybrid RAG: vector + BM25 retrieval, cross-encoder re-ranking, "
        "cited answers with confidence-based abstention."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router)
app.include_router(query.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/config")
def get_config() -> dict:
    """Retrieval defaults, so the UI can initialise its controls from the server."""
    return {
        "hybrid_alpha": settings.hybrid_alpha,
        "top_k": settings.top_k,
        "confidence_threshold": settings.confidence_threshold,
        "vector_confidence_threshold": settings.vector_confidence_threshold,
        "candidate_pool": settings.candidate_pool,
        "rerank_pool": settings.rerank_pool,
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model,
        "claude_model": settings.claude_model,
    }
