import re


# Rough approximation: 1 token ≈ 4 characters (good enough for chunking purposes)
CHARS_PER_TOKEN = 4


def chunk_pages(
    pages: list[dict],
    file_id: str,
    file_name: str,
    max_tokens: int = 400,
    overlap_paragraphs: int = 1,
) -> list[dict]:
    """
    Chunk a list of {page_number, text} dicts into overlapping text chunks.

    Returns list of chunk dicts with keys:
        chunk_id, file_id, file_name, page_number, chunk_index, text
    """
    max_chars = max_tokens * CHARS_PER_TOKEN
    chunks = []
    chunk_index = 0

    for page in pages:
        page_number = page["page_number"]
        paragraphs = _split_paragraphs(page["text"])

        current_paragraphs: list[str] = []
        current_chars = 0

        for para in paragraphs:
            para_chars = len(para)

            if current_chars + para_chars > max_chars and current_paragraphs:
                chunk = _make_chunk(
                    current_paragraphs, chunk_index, file_id, file_name, page_number
                )
                chunks.append(chunk)
                chunk_index += 1

                # Carry over the last N paragraphs for overlap
                current_paragraphs = current_paragraphs[-overlap_paragraphs:] if overlap_paragraphs else []
                current_chars = sum(len(p) for p in current_paragraphs)

            current_paragraphs.append(para)
            current_chars += para_chars

        if current_paragraphs:
            chunk = _make_chunk(
                current_paragraphs, chunk_index, file_id, file_name, page_number
            )
            chunks.append(chunk)
            chunk_index += 1

    return chunks


def _split_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs on double newlines; fall back to sentences for dense text."""
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if len(paragraphs) <= 1 and len(text) > 800:
        # Dense text with no double newlines — split on sentence endings
        sentences = re.split(r"(?<=[.!?])\s+", text)
        paragraphs = [s.strip() for s in sentences if s.strip()]
    return paragraphs


def _make_chunk(
    paragraphs: list[str],
    chunk_index: int,
    file_id: str,
    file_name: str,
    page_number: int,
) -> dict:
    return {
        "chunk_id": f"{file_id}_{page_number}_{chunk_index}",
        "file_id": file_id,
        "file_name": file_name,
        "page_number": page_number,
        "chunk_index": chunk_index,
        "text": "\n\n".join(paragraphs),
    }
