"""Text extraction and page-aware chunking.

Citations need to name a page, but chunking page-by-page would truncate every
chunk at each page boundary and lose sentences that straddle two pages. So we
do the opposite: concatenate all pages into one string while recording where
each page starts, split the whole thing normally, then map each chunk's start
offset back to a page with a binary search.

``RecursiveCharacterTextSplitter(add_start_index=True)`` gives us the offset for
free -- LangChain records it on each produced Document.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.config import settings


@dataclass
class Chunk:
    text: str
    chunk_index: int
    page: int
    char_start: int


@dataclass
class ExtractedText:
    text: str
    # Character offset at which each page begins, ascending. page N (1-indexed)
    # starts at page_starts[N - 1].
    page_starts: list[int]
    num_pages: int


PAGE_SEPARATOR = "\n\n"


def extract_text(file_path: Path) -> ExtractedText:
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(str(file_path))
        parts: list[str] = []
        page_starts: list[int] = []
        cursor = 0
        for page in reader.pages:
            page_starts.append(cursor)
            page_text = page.extract_text() or ""
            parts.append(page_text)
            cursor += len(page_text) + len(PAGE_SEPARATOR)
        return ExtractedText(
            text=PAGE_SEPARATOR.join(parts),
            page_starts=page_starts,
            num_pages=len(reader.pages),
        )

    if suffix in (".txt", ".md"):
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        # Plain text has no pages; treat the whole file as page 1.
        return ExtractedText(text=text, page_starts=[0], num_pages=1)

    raise ValueError(f"Unsupported file type: {suffix}")


def _page_for_offset(offset: int, page_starts: list[int]) -> int:
    """Map a character offset to a 1-indexed page number."""
    # bisect_right - 1 gives the index of the last page starting at or before
    # this offset.
    return max(1, bisect.bisect_right(page_starts, offset))


def chunk_text(extracted: ExtractedText) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    documents = splitter.create_documents([extracted.text])

    chunks: list[Chunk] = []
    for doc in documents:
        text = doc.page_content.strip()
        if not text:
            continue
        start = doc.metadata.get("start_index", 0)
        chunks.append(
            Chunk(
                text=text,
                chunk_index=len(chunks),
                page=_page_for_offset(start, extracted.page_starts),
                char_start=start,
            )
        )
    return chunks


def process_file(file_path: Path) -> tuple[list[Chunk], int]:
    """Extract, chunk, and return (chunks, page_count)."""
    extracted = extract_text(file_path)
    if not extracted.text.strip():
        raise ValueError(
            "No extractable text found in document. If this is a scanned PDF, "
            "it needs OCR before it can be indexed."
        )
    return chunk_text(extracted), extracted.num_pages
