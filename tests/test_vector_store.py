import pytest

from src.store.vector_store import (
    upsert_chunks,
    query_chunks,
    delete_by_file_id,
    list_papers,
    list_indexed_files,
    COLLECTION_NAME,
)


class TestUpsertChunks:
    def test_returns_chunk_ids(self, qdrant_client, mock_embedder, sample_chunks):
        # Act
        ids = upsert_chunks(sample_chunks, client=qdrant_client, embedder=mock_embedder)

        # Assert
        assert set(ids) == {c["chunk_id"] for c in sample_chunks}

    def test_empty_input_returns_empty_list(self, qdrant_client, mock_embedder):
        # Act
        ids = upsert_chunks([], client=qdrant_client, embedder=mock_embedder)

        # Assert
        assert ids == []

    def test_upsert_is_idempotent(self, qdrant_client, mock_embedder, sample_chunks):
        # Act — upsert twice
        upsert_chunks(sample_chunks, client=qdrant_client, embedder=mock_embedder)
        upsert_chunks(sample_chunks, client=qdrant_client, embedder=mock_embedder)

        # Assert — collection has exactly len(sample_chunks) items
        count = qdrant_client.count(COLLECTION_NAME).count
        assert count == len(sample_chunks)


class TestQueryChunks:
    def test_returns_expected_fields(self, qdrant_client, mock_embedder, sample_chunks):
        # Arrange
        upsert_chunks(sample_chunks, client=qdrant_client, embedder=mock_embedder)

        # Act
        results = query_chunks("climate", top_k=2, client=qdrant_client, embedder=mock_embedder)

        # Assert
        assert len(results) <= 2
        for r in results:
            assert "chunk_id" in r
            assert "file_name" in r
            assert "page_number" in r
            assert "text" in r
            assert "score" in r

    def test_top_k_limits_results(self, qdrant_client, mock_embedder, sample_chunks):
        # Arrange
        upsert_chunks(sample_chunks, client=qdrant_client, embedder=mock_embedder)

        # Act
        results = query_chunks("research", top_k=1, client=qdrant_client, embedder=mock_embedder)

        # Assert
        assert len(results) <= 1


class TestDeleteByFileId:
    def test_deletes_only_target_file_chunks(self, qdrant_client, mock_embedder, sample_chunks):
        # Arrange
        upsert_chunks(sample_chunks, client=qdrant_client, embedder=mock_embedder)

        # Act
        delete_by_file_id("file1", client=qdrant_client)

        # Assert
        points, _ = qdrant_client.scroll(COLLECTION_NAME, limit=100, with_payload=True, with_vectors=False)
        file_ids = [p.payload["file_id"] for p in points]
        assert "file1" not in file_ids
        assert "file2" in file_ids

    def test_delete_nonexistent_file_is_safe(self, qdrant_client, mock_embedder, sample_chunks):
        # Arrange
        upsert_chunks(sample_chunks, client=qdrant_client, embedder=mock_embedder)

        # Act & Assert — should not raise
        delete_by_file_id("nonexistent_id", client=qdrant_client)
        count = qdrant_client.count(COLLECTION_NAME).count
        assert count == len(sample_chunks)


class TestListPapers:
    def test_returns_unique_papers(self, qdrant_client, mock_embedder, sample_chunks):
        # Arrange
        upsert_chunks(sample_chunks, client=qdrant_client, embedder=mock_embedder)

        # Act
        papers = list_papers(client=qdrant_client)

        # Assert
        assert len(papers) == 2  # file1 and file2
        file_ids = {p["file_id"] for p in papers}
        assert file_ids == {"file1", "file2"}

    def test_chunk_count_is_correct(self, qdrant_client, mock_embedder, sample_chunks):
        # Arrange
        upsert_chunks(sample_chunks, client=qdrant_client, embedder=mock_embedder)

        # Act
        papers = list_papers(client=qdrant_client)
        counts = {p["file_id"]: p["chunk_count"] for p in papers}

        # Assert
        assert counts["file1"] == 2
        assert counts["file2"] == 1


class TestListIndexedFiles:
    def test_returns_file_id_to_modified_time_mapping(self, qdrant_client, mock_embedder):
        # Arrange
        chunks = [
            {
                "chunk_id": "f1_1_0", "file_id": "f1", "file_name": "a.pdf",
                "page_number": 1, "chunk_index": 0, "text": "hello", "modified_time": "2024-01-01",
            },
            {
                "chunk_id": "f1_1_1", "file_id": "f1", "file_name": "a.pdf",
                "page_number": 1, "chunk_index": 1, "text": "world", "modified_time": "2024-01-01",
            },
            {
                "chunk_id": "f2_1_0", "file_id": "f2", "file_name": "b.pdf",
                "page_number": 1, "chunk_index": 0, "text": "other", "modified_time": "2024-06-01",
            },
        ]
        upsert_chunks(chunks, client=qdrant_client, embedder=mock_embedder)

        # Act
        result = list_indexed_files(client=qdrant_client)

        # Assert
        assert result == {"f1": "2024-01-01", "f2": "2024-06-01"}

    def test_returns_empty_dict_when_collection_empty(self, qdrant_client):
        # Act
        result = list_indexed_files(client=qdrant_client)

        # Assert
        assert result == {}
