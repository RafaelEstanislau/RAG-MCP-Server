from pathlib import Path
from unittest.mock import MagicMock

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_PDF = FIXTURES_DIR / "sample.pdf"
SAMPLE_DOCX = FIXTURES_DIR / "sample.docx"


@pytest.fixture
def qdrant_client() -> QdrantClient:
    """Real in-memory Qdrant client — no disk I/O, isolated per test."""
    client = QdrantClient(":memory:")
    client.create_collection(
        "references",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
    client.create_payload_index(
        collection_name="references",
        field_name="file_id",
        field_schema="keyword",
    )
    return client


@pytest.fixture
def mock_embedder():
    """Deterministic fake embedder — returns 384-dim vectors without loading any model."""
    def _embed(texts: list[str]) -> list[list[float]]:
        return [[0.1] * 384 for _ in texts]
    return _embed


@pytest.fixture
def mock_drive_service():
    """MagicMock mimicking the Google Drive API service object."""
    return MagicMock()


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
