import pytest

from src.ingestion.chunker import chunk_pages, _split_paragraphs, CHARS_PER_TOKEN


FILE_ID = "testfile"
FILE_NAME = "test.pdf"


def _make_pages(texts: list[str]) -> list[dict]:
    return [{"page_number": i + 1, "text": t} for i, t in enumerate(texts)]


class TestChunkPages:
    def test_returns_list_of_chunk_dicts(self):
        # Arrange
        pages = _make_pages(["First paragraph.\n\nSecond paragraph."])

        # Act
        chunks = chunk_pages(pages, file_id=FILE_ID, file_name=FILE_NAME)

        # Assert
        assert isinstance(chunks, list)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert "chunk_id" in chunk
            assert "file_id" in chunk
            assert "file_name" in chunk
            assert "page_number" in chunk
            assert "chunk_index" in chunk
            assert "text" in chunk

    def test_chunk_id_format(self):
        # Arrange
        pages = _make_pages(["A paragraph."])

        # Act
        chunks = chunk_pages(pages, file_id=FILE_ID, file_name=FILE_NAME)

        # Assert
        assert chunks[0]["chunk_id"] == f"{FILE_ID}_1_0"

    def test_no_chunk_exceeds_token_limit(self):
        # Arrange — many short paragraphs each well under the limit but together exceeding it
        max_tokens = 20  # 80 chars
        # Each paragraph is ~40 chars, so two fit but three do not
        para = "Short sentence here. Another one follows."  # ~42 chars
        text = "\n\n".join([para] * 10)
        pages = _make_pages([text])

        # Act
        chunks = chunk_pages(pages, file_id=FILE_ID, file_name=FILE_NAME, max_tokens=max_tokens)

        # Assert — each chunk must be at or near the limit (within one paragraph overshoot)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk["text"]) <= max_tokens * CHARS_PER_TOKEN + len(para)

    def test_overlap_paragraph_appears_in_consecutive_chunks(self):
        # Arrange — two paragraphs that together exceed the limit
        para1 = "Alpha. " * 80   # ~560 chars
        para2 = "Beta. " * 80    # ~480 chars
        para3 = "Gamma. " * 80
        text = f"{para1}\n\n{para2}\n\n{para3}"
        pages = _make_pages([text])

        # Act
        chunks = chunk_pages(
            pages,
            file_id=FILE_ID,
            file_name=FILE_NAME,
            max_tokens=100,
            overlap_paragraphs=1,
        )

        # Assert — with overlap, more than 1 chunk, and consecutive chunks share a paragraph
        assert len(chunks) >= 2
        # The last paragraph of chunk 0 should appear at the start of chunk 1
        chunk0_paras = chunks[0]["text"].split("\n\n")
        chunk1_paras = chunks[1]["text"].split("\n\n")
        assert chunk0_paras[-1] == chunk1_paras[0]

    def test_metadata_fields_populated(self):
        # Arrange
        pages = _make_pages(["Simple paragraph text."])

        # Act
        chunks = chunk_pages(pages, file_id=FILE_ID, file_name=FILE_NAME)

        # Assert
        chunk = chunks[0]
        assert chunk["file_id"] == FILE_ID
        assert chunk["file_name"] == FILE_NAME
        assert chunk["page_number"] == 1

    def test_empty_pages_returns_empty_list(self):
        # Act
        chunks = chunk_pages([], file_id=FILE_ID, file_name=FILE_NAME)

        # Assert
        assert chunks == []

    def test_multiple_pages_produce_distinct_chunks(self):
        # Arrange
        pages = _make_pages(["Page one content.", "Page two content."])

        # Act
        chunks = chunk_pages(pages, file_id=FILE_ID, file_name=FILE_NAME)

        # Assert
        assert len(chunks) == 2
        assert chunks[0]["page_number"] == 1
        assert chunks[1]["page_number"] == 2


class TestSplitParagraphs:
    def test_splits_on_double_newline(self):
        # Arrange
        text = "First para.\n\nSecond para.\n\nThird para."

        # Act
        result = _split_paragraphs(text)

        # Assert
        assert result == ["First para.", "Second para.", "Third para."]

    def test_filters_empty_paragraphs(self):
        # Arrange
        text = "First.\n\n\n\nSecond."

        # Act
        result = _split_paragraphs(text)

        # Assert
        assert "" not in result
        assert len(result) == 2
