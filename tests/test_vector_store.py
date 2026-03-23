import pytest

from src.store.vector_store import upsert_chunks, query_chunks, delete_by_file_id, list_papers


class TestUpsertChunks:
    def test_returns_chunk_ids(self, ephemeral_chroma, mock_embedder, sample_chunks):
        # Act
        ids = upsert_chunks(sample_chunks, client=ephemeral_chroma, embedder=mock_embedder)

        # Assert
        assert set(ids) == {c["chunk_id"] for c in sample_chunks}

    def test_empty_input_returns_empty_list(self, ephemeral_chroma, mock_embedder):
        # Act
        ids = upsert_chunks([], client=ephemeral_chroma, embedder=mock_embedder)

        # Assert
        assert ids == []

    def test_upsert_is_idempotent(self, ephemeral_chroma, mock_embedder, sample_chunks):
        # Act — upsert twice
        upsert_chunks(sample_chunks, client=ephemeral_chroma, embedder=mock_embedder)
        upsert_chunks(sample_chunks, client=ephemeral_chroma, embedder=mock_embedder)

        # Assert — collection has exactly len(sample_chunks) items
        collection = ephemeral_chroma.get_or_create_collection("references")
        assert collection.count() == len(sample_chunks)


class TestQueryChunks:
    def test_returns_expected_fields(self, ephemeral_chroma, mock_embedder, sample_chunks):
        # Arrange
        upsert_chunks(sample_chunks, client=ephemeral_chroma, embedder=mock_embedder)

        # Act
        results = query_chunks("climate", top_k=2, client=ephemeral_chroma, embedder=mock_embedder)

        # Assert
        assert len(results) <= 2
        for r in results:
            assert "chunk_id" in r
            assert "file_name" in r
            assert "page_number" in r
            assert "text" in r
            assert "score" in r

    def test_top_k_limits_results(self, ephemeral_chroma, mock_embedder, sample_chunks):
        # Arrange
        upsert_chunks(sample_chunks, client=ephemeral_chroma, embedder=mock_embedder)

        # Act
        results = query_chunks("research", top_k=1, client=ephemeral_chroma, embedder=mock_embedder)

        # Assert
        assert len(results) <= 1



class TestDeleteByFileId:
    def test_deletes_only_target_file_chunks(self, ephemeral_chroma, mock_embedder, sample_chunks):
        # Arrange
        upsert_chunks(sample_chunks, client=ephemeral_chroma, embedder=mock_embedder)

        # Act
        delete_by_file_id("file1", client=ephemeral_chroma)

        # Assert
        collection = ephemeral_chroma.get_or_create_collection("references")
        remaining = collection.get(include=["metadatas"])
        file_ids = [m["file_id"] for m in remaining["metadatas"]]
        assert "file1" not in file_ids
        assert "file2" in file_ids

    def test_delete_nonexistent_file_is_safe(self, ephemeral_chroma, mock_embedder, sample_chunks):
        # Arrange
        upsert_chunks(sample_chunks, client=ephemeral_chroma, embedder=mock_embedder)

        # Act & Assert — should not raise
        delete_by_file_id("nonexistent_id", client=ephemeral_chroma)
        collection = ephemeral_chroma.get_or_create_collection("references")
        assert collection.count() == len(sample_chunks)


class TestListPapers:
    def test_returns_unique_papers(self, ephemeral_chroma, mock_embedder, sample_chunks):
        # Arrange
        upsert_chunks(sample_chunks, client=ephemeral_chroma, embedder=mock_embedder)

        # Act
        papers = list_papers(client=ephemeral_chroma)

        # Assert
        assert len(papers) == 2  # file1 and file2
        file_ids = {p["file_id"] for p in papers}
        assert file_ids == {"file1", "file2"}

    def test_chunk_count_is_correct(self, ephemeral_chroma, mock_embedder, sample_chunks):
        # Arrange
        upsert_chunks(sample_chunks, client=ephemeral_chroma, embedder=mock_embedder)

        # Act
        papers = list_papers(client=ephemeral_chroma)
        counts = {p["file_id"]: p["chunk_count"] for p in papers}

        # Assert
        assert counts["file1"] == 2
        assert counts["file2"] == 1

