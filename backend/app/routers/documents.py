import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile

from app.bm25_index import refresh_bm25_index
from app.config import settings
from app.document_processor import process_file
from app.embeddings import embed_texts
from app.schemas import DocumentInfo, DocumentListResponse, UploadResponse
from app.vector_store import add_chunks, delete_document, list_documents

router = APIRouter(prefix="/documents", tags=["documents"])

ALLOWED_SUFFIXES = {".pdf", ".txt", ".md"}


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile) -> UploadResponse:
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Accepted: PDF, TXT, MD.",
        )

    document_id = uuid.uuid4().hex
    dest_path = settings.upload_path / f"{document_id}{suffix}"
    dest_path.write_bytes(await file.read())

    try:
        chunks, num_pages = process_file(dest_path)
    except ValueError as exc:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    embeddings = embed_texts([chunk.text for chunk in chunks])
    add_chunks(document_id, file.filename, chunks, embeddings, num_pages)

    # BM25 has no incremental-update path -- IDF depends on the whole corpus,
    # so adding a document changes the scores of every existing chunk.
    refresh_bm25_index()

    return UploadResponse(
        document=DocumentInfo(
            document_id=document_id,
            filename=file.filename,
            num_chunks=len(chunks),
            num_pages=num_pages,
        ),
        message=f"Indexed {len(chunks)} chunks across {num_pages} page(s).",
    )


@router.get("", response_model=DocumentListResponse)
def get_documents() -> DocumentListResponse:
    return DocumentListResponse(
        documents=[DocumentInfo(**doc) for doc in list_documents()]
    )


@router.delete("/{document_id}")
def remove_document(document_id: str) -> dict:
    delete_document(document_id)
    for path in settings.upload_path.glob(f"{document_id}.*"):
        path.unlink(missing_ok=True)
    refresh_bm25_index()
    return {"message": f"Document {document_id} deleted"}
