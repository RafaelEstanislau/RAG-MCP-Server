from pathlib import Path
from unittest.mock import MagicMock

import chromadb
import pytest


FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_PDF = FIXTURES_DIR / "sample.pdf"
SAMPLE_DOCX = FIXTURES_DIR / "sample.docx"


@pytest.fixture
def ephemeral_chroma() -> chromadb.EphemeralClient:
    """Real in-memory ChromaDB client — no disk I/O, isolated per test."""
    return chromadb.EphemeralClient()


@pytest.fixture
def mock_embedder():
    """Deterministic fake embedder — returns 384-dim zero vectors without loading any model."""
    def _embed(texts: list[str]) -> list[list[float]]:
        return [[0.1] * 384 for _ in texts]
    return _embed


@pytest.fixture
def mock_drive_service():
    """MagicMock mimicking the Google Drive API service object."""
    service = MagicMock()
    return service


@pytest.fixture
def sample_chunks() -> list[dict]:
    return [
        {
            "chunk_id": "file1_1_0",
            "file_id": "file1",
            "file_name": "paper_one.pdf",
            "page_number": 1,
            "chunk_index": 0,
            "text": "This paper discusses climate change impacts on biodiversity.",
        },
        {
            "chunk_id": "file1_1_1",
            "file_id": "file1",
            "file_name": "paper_one.pdf",
            "page_number": 1,
            "chunk_index": 1,
            "text": "Methodology: We used a mixed-methods approach combining surveys and interviews.",
        },
        {
            "chunk_id": "file2_1_0",
            "file_id": "file2",
            "file_name": "paper_two.pdf",
            "page_number": 1,
            "chunk_index": 0,
            "text": "Results show a 30% increase in species migration over the past decade.",
        },
    ]
