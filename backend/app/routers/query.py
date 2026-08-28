import httpx
from fastapi import APIRouter, HTTPException

from app.config import settings
from app.rag_service import generate_answer
from app.schemas import QueryRequest, QueryResponse

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty")

    try:
        answer, sources, abstained, stats = generate_answer(
            request.question,
            top_k=request.top_k,
            document_id=request.document_id,
            alpha=request.alpha,
            use_reranker=request.use_reranker,
        )
    except httpx.ConnectError as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                f"Could not reach Ollama at {settings.ollama_url}. "
                "Is it running? Try `ollama serve`."
            ),
        ) from exc
    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=504,
            detail="Ollama did not respond in time.",
        ) from exc
    except httpx.HTTPStatusError as exc:
        detail = f"Ollama returned {exc.response.status_code}."
        if exc.response.status_code == 404:
            detail += (
                f" Model '{settings.ollama_model}' may not be pulled -- "
                f"try `ollama pull {settings.ollama_model}`."
            )
        raise HTTPException(status_code=502, detail=detail) from exc

    return QueryResponse(
        answer=answer, sources=sources, abstained=abstained, stats=stats
    )
