"""ChromaDB persistence layer.

Chroma is the source of truth for chunk text and metadata. The BM25 index is
rebuilt from here on startup and after every mutation, so there is exactly one
place a chunk can live and the two retrievers can never drift apart.
"""

from functools import lru_cache

import chromadb

from app.config import settings
from app.document_processor import Chunk


@lru_cache(maxsize=1)
def get_collection():
    client = chromadb.PersistentClient(path=str(settings.chroma_path))
    return client.get_or_create_collection(
        name=settings.chroma_collection,
        # Cosine, to match the normalized sentence-transformers embeddings.
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks(
    document_id: str,
    filename: str,
    chunks: list[Chunk],
    embeddings: list[list[float]],
    num_pages: int,
) -> None:
    collection = get_collection()
    collection.add(
        ids=[f"{document_id}_{c.chunk_index}" for c in chunks],
        embeddings=embeddings,
        documents=[c.text for c in chunks],
        metadatas=[
            {
                "document_id": document_id,
                "filename": filename,
                "chunk_index": c.chunk_index,
                "page": c.page,
                "num_pages": num_pages,
            }
            for c in chunks
        ],
    )


def query_chunks(
    query_embedding: list[float], top_k: int, document_id: str | None = None
) -> dict:
    collection = get_collection()
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"document_id": document_id} if document_id else None,
    )


def get_all_chunks() -> list[dict]:
    """Every chunk with its text and metadata -- used to build the BM25 index."""
    collection = get_collection()
    data = collection.get(include=["documents", "metadatas"])
    return [
        {"id": chunk_id, "text": text, "metadata": metadata}
        for chunk_id, text, metadata in zip(
            data["ids"], data["documents"], data["metadatas"]
        )
    ]


def list_documents() -> list[dict]:
    collection = get_collection()
    data = collection.get(include=["metadatas"])
    documents: dict[str, dict] = {}
    for metadata in data["metadatas"]:
        doc_id = metadata["document_id"]
        if doc_id not in documents:
            documents[doc_id] = {
                "document_id": doc_id,
                "filename": metadata["filename"],
                "num_chunks": 0,
                "num_pages": metadata.get("num_pages", 1),
            }
        documents[doc_id]["num_chunks"] += 1
    return list(documents.values())


def delete_document(document_id: str) -> None:
    get_collection().delete(where={"document_id": document_id})


def reset_collection() -> None:
    """Drop every chunk. Used by the eval harness between runs."""
    client = chromadb.PersistentClient(path=str(settings.chroma_path))
    try:
        client.delete_collection(settings.chroma_collection)
    except Exception:
        pass
    get_collection.cache_clear()
