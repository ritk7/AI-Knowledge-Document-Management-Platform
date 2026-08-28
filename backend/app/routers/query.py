import anthropic
from fastapi import APIRouter, HTTPException

from app.config import settings
from app.rag_service import generate_answer
from app.schemas import QueryRequest, QueryResponse

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty")
    if not settings.anthropic_api_key:
        raise HTTPException(
            status_code=500,
            detail="ANTHROPIC_API_KEY is not configured on the server.",
        )

    try:
        answer, sources, abstained, stats = generate_answer(
            request.question,
            top_k=request.top_k,
            document_id=request.document_id,
            alpha=request.alpha,
            use_reranker=request.use_reranker,
        )
    except anthropic.AuthenticationError as exc:
        raise HTTPException(
            status_code=502, detail="Anthropic rejected the API key."
        ) from exc
    except anthropic.RateLimitError as exc:
        raise HTTPException(
            status_code=429, detail="Rate limited by the Anthropic API. Retry shortly."
        ) from exc
    except anthropic.APIStatusError as exc:
        raise HTTPException(
            status_code=502, detail=f"Anthropic API error ({exc.status_code})."
        ) from exc
    except anthropic.APIConnectionError as exc:
        raise HTTPException(
            status_code=502, detail="Could not reach the Anthropic API."
        ) from exc

    return QueryResponse(
        answer=answer, sources=sources, abstained=abstained, stats=stats
    )
