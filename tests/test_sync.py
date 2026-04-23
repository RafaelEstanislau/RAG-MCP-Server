from unittest.mock import MagicMock, patch

import pytest

from src.drive import sync as sync_module


class TestSyncDrive:
    def _make_file_entry(self, file_id, name, mime="application/pdf", modified="2024-01-01T00:00:00.000Z"):
        return {"id": file_id, "name": name, "mimeType": mime, "modifiedTime": modified}

    def test_new_file_is_indexed(self, monkeypatch):
        # Arrange
        monkeypatch.setattr(sync_module.settings, "drive_folder_id", "folder1")
        file_entry = self._make_file_entry("f1", "paper.pdf")

        with (
            patch("src.drive.sync._build_drive_service", return_value=MagicMock()),
            patch("src.drive.sync._list_drive_files", return_value=[file_entry]),
            patch("src.drive.sync.list_indexed_files", return_value={}),
            patch("src.drive.sync._download_file", return_value=b"pdf bytes"),
            patch("src.drive.sync._extract", return_value=[{"page_number": 1, "text": "Content."}]),
            patch("src.drive.sync.chunk_pages", return_value=[
                {"chunk_id": "f1_1_0", "file_id": "f1", "file_name": "paper.pdf",
                 "page_number": 1, "chunk_index": 0, "text": "Content."}
            ]),
            patch("src.drive.sync.upsert_chunks", return_value=["f1_1_0"]),
            patch("src.drive.sync.delete_by_file_id") as mock_delete,
        ):
            from src.drive.sync import sync_drive
            result = sync_drive()

        # Assert
        assert result["added"] == 1
        assert result["skipped"] == 0
        mock_delete.assert_not_called()

    def test_unchanged_file_is_skipped(self, monkeypatch):
        # Arrange
        monkeypatch.setattr(sync_module.settings, "drive_folder_id", "folder1")
        modified = "2024-01-01T00:00:00.000Z"
        file_entry = self._make_file_entry("f1", "paper.pdf", modified=modified)

        with (
            patch("src.drive.sync._build_drive_service", return_value=MagicMock()),
            patch("src.drive.sync._list_drive_files", return_value=[file_entry]),
            patch("src.drive.sync.list_indexed_files", return_value={"f1": modified}),
            patch("src.drive.sync.upsert_chunks") as mock_upsert,
        ):
            from src.drive.sync import sync_drive
            result = sync_drive()

        # Assert
        assert result["skipped"] == 1
        mock_upsert.assert_not_called()

    def test_updated_file_deletes_old_chunks_then_reindexes(self, monkeypatch):
        # Arrange
        monkeypatch.setattr(sync_module.settings, "drive_folder_id", "folder1")
        old_modified = "2024-01-01T00:00:00.000Z"
        new_modified = "2024-06-01T00:00:00.000Z"
        file_entry = self._make_file_entry("f1", "paper.pdf", modified=new_modified)

        with (
            patch("src.drive.sync._build_drive_service", return_value=MagicMock()),
            patch("src.drive.sync._list_drive_files", return_value=[file_entry]),
            patch("src.drive.sync.list_indexed_files", return_value={"f1": old_modified}),
            patch("src.drive.sync._download_file", return_value=b"pdf bytes"),
            patch("src.drive.sync._extract", return_value=[{"page_number": 1, "text": "Updated content."}]),
            patch("src.drive.sync.chunk_pages", return_value=[
                {"chunk_id": "f1_1_0", "file_id": "f1", "file_name": "paper.pdf",
                 "page_number": 1, "chunk_index": 0, "text": "Updated content."}
            ]),
            patch("src.drive.sync.upsert_chunks", return_value=["f1_1_0"]),
            patch("src.drive.sync.delete_by_file_id") as mock_delete,
        ):
            from src.drive.sync import sync_drive
            result = sync_drive()

        # Assert
        assert result["updated"] == 1
        mock_delete.assert_called_once_with("f1")

    def test_deleted_file_is_removed_from_index(self, monkeypatch):
        # Arrange — Drive has 0 files, but index still has f1
        monkeypatch.setattr(sync_module.settings, "drive_folder_id", "folder1")

        with (
            patch("src.drive.sync._build_drive_service", return_value=MagicMock()),
            patch("src.drive.sync._list_drive_files", return_value=[]),
            patch("src.drive.sync.list_indexed_files", return_value={"f1": "2024-01-01T00:00:00.000Z"}),
            patch("src.drive.sync.delete_by_file_id") as mock_delete,
            patch("src.drive.sync.upsert_chunks") as mock_upsert,
        ):
            from src.drive.sync import sync_drive
            result = sync_drive()

        # Assert
        assert result["removed"] == 1
        assert result["total"] == 0
        mock_delete.assert_called_once_with("f1")
        mock_upsert.assert_not_called()

    def test_chunks_stamped_with_modified_time(self, monkeypatch):
        # Arrange
        monkeypatch.setattr(sync_module.settings, "drive_folder_id", "folder1")
        modified = "2024-03-01T00:00:00.000Z"
        file_entry = self._make_file_entry("f1", "paper.pdf", modified=modified)
        chunks = [
            {"chunk_id": "f1_1_0", "file_id": "f1", "file_name": "paper.pdf",
             "page_number": 1, "chunk_index": 0, "text": "Content."}
        ]

        with (
            patch("src.drive.sync._build_drive_service", return_value=MagicMock()),
            patch("src.drive.sync._list_drive_files", return_value=[file_entry]),
            patch("src.drive.sync.list_indexed_files", return_value={}),
            patch("src.drive.sync._download_file", return_value=b"pdf bytes"),
            patch("src.drive.sync._extract", return_value=[{"page_number": 1, "text": "Content."}]),
            patch("src.drive.sync.chunk_pages", return_value=chunks),
            patch("src.drive.sync.upsert_chunks") as mock_upsert,
            patch("src.drive.sync.delete_by_file_id"),
        ):
            from src.drive.sync import sync_drive
            sync_drive()

        # Assert
        called_chunks = mock_upsert.call_args[0][0]
        assert called_chunks[0]["modified_time"] == modified
