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

    # Checked before chunking/embedding, not after: with no cap, a 34MB file
    # took 58 minutes to embed (42,857 chunks) and still failed at the final
    # write because it exceeded ChromaDB's own batch limit. Rejecting here
    # costs milliseconds instead of the better part of an hour.
    contents = await file.read()
    max_bytes = int(settings.max_upload_mb * 1024 * 1024)
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File is {len(contents) / 1024 / 1024:.1f}MB, which exceeds "
                f"the {settings.max_upload_mb:.0f}MB limit."
            ),
        )

    document_id = uuid.uuid4().hex
    dest_path = settings.upload_path / f"{document_id}{suffix}"
    dest_path.write_bytes(contents)

    # Everything from here through add_chunks() is one failure unit: a crash
    # partway through -- bad PDF, an embedding-model error, a ChromaDB write
    # failure -- must not leave the uploaded file or a partially-indexed
    # document behind. Verified live: before this covered embed_texts() and
    # add_chunks() too (not just process_file()), an unbatched Chroma write
    # that exceeded its internal batch-size limit crashed with a bare 500 and
    # left the uploaded file orphaned on disk permanently, since the cleanup
    # path never ran for exceptions raised after process_file() returned.
    try:
        chunks, num_pages = process_file(dest_path)
        embeddings = embed_texts([chunk.text for chunk in chunks])
        add_chunks(document_id, file.filename, chunks, embeddings, num_pages)
    except Exception as exc:
        dest_path.unlink(missing_ok=True)
        # add_chunks() batches its ChromaDB writes; a failure partway through
        # can leave earlier batches already committed. delete_document() is a
        # filter-delete that is a no-op if nothing was written, so it is safe
        # to call unconditionally as part of cleanup.
        delete_document(document_id)
        if isinstance(exc, ValueError):
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        raise HTTPException(
            status_code=500, detail="Failed to index document."
        ) from exc

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
