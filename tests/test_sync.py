import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from src.drive import sync as sync_module
from src.drive.sync import _load_sync_state, _save_sync_state


class TestSyncDrive:
    def _make_file_entry(self, file_id, name, mime="application/pdf", modified="2024-01-01T00:00:00.000Z"):
        return {"id": file_id, "name": name, "mimeType": mime, "modifiedTime": modified}

    def test_new_file_is_indexed(self, tmp_path, monkeypatch, mock_embedder):
        # Arrange
        monkeypatch.setattr(sync_module.settings, "sync_state_path", tmp_path / "state.json")
        monkeypatch.setattr(sync_module.settings, "downloads_path", tmp_path / "downloads")
        monkeypatch.setattr(sync_module.settings, "drive_folder_id", "folder1")

        file_entry = self._make_file_entry("f1", "paper.pdf")

        mock_service = MagicMock()
        mock_service.files().list().execute.return_value = {
            "files": [file_entry],
            "nextPageToken": None,
        }
        # Simulate file download — write dummy bytes
        def fake_get_media(fileId):
            req = MagicMock()
            return req

        mock_service.files().get_media.side_effect = fake_get_media

        dummy_pdf_bytes = b"%PDF-1.4 dummy"

        with (
            patch("src.drive.sync._build_drive_service", return_value=mock_service),
            patch("src.drive.sync._download_file", return_value=tmp_path / "f1.pdf"),
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

    def test_unchanged_file_is_skipped(self, tmp_path, monkeypatch):
        # Arrange
        state_path = tmp_path / "state.json"
        monkeypatch.setattr(sync_module.settings, "sync_state_path", state_path)
        monkeypatch.setattr(sync_module.settings, "downloads_path", tmp_path / "downloads")
        monkeypatch.setattr(sync_module.settings, "drive_folder_id", "folder1")

        modified = "2024-01-01T00:00:00.000Z"
        state_path.write_text(json.dumps({
            "f1": {"name": "paper.pdf", "modified_time": modified, "chunk_ids": ["f1_1_0"]}
        }))

        file_entry = self._make_file_entry("f1", "paper.pdf", modified=modified)
        mock_service = MagicMock()
        mock_service.files().list().execute.return_value = {
            "files": [file_entry],
            "nextPageToken": None,
        }

        with (
            patch("src.drive.sync._build_drive_service", return_value=mock_service),
            patch("src.drive.sync.upsert_chunks") as mock_upsert,
        ):
            from src.drive.sync import sync_drive
            result = sync_drive()

        # Assert
        assert result["skipped"] == 1
        mock_upsert.assert_not_called()

    def test_updated_file_deletes_old_chunks_then_reindexes(self, tmp_path, monkeypatch):
        # Arrange
        state_path = tmp_path / "state.json"
        monkeypatch.setattr(sync_module.settings, "sync_state_path", state_path)
        monkeypatch.setattr(sync_module.settings, "downloads_path", tmp_path / "downloads")
        monkeypatch.setattr(sync_module.settings, "drive_folder_id", "folder1")

        old_modified = "2024-01-01T00:00:00.000Z"
        new_modified = "2024-06-01T00:00:00.000Z"
        state_path.write_text(json.dumps({
            "f1": {"name": "paper.pdf", "modified_time": old_modified, "chunk_ids": ["f1_1_0"]}
        }))

        file_entry = self._make_file_entry("f1", "paper.pdf", modified=new_modified)
        mock_service = MagicMock()
        mock_service.files().list().execute.return_value = {
            "files": [file_entry],
            "nextPageToken": None,
        }

        with (
            patch("src.drive.sync._build_drive_service", return_value=mock_service),
            patch("src.drive.sync._download_file", return_value=tmp_path / "f1.pdf"),
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

    def test_sync_state_updated_after_indexing(self, tmp_path, monkeypatch):
        # Arrange
        state_path = tmp_path / "state.json"
        monkeypatch.setattr(sync_module.settings, "sync_state_path", state_path)
        monkeypatch.setattr(sync_module.settings, "downloads_path", tmp_path / "downloads")
        monkeypatch.setattr(sync_module.settings, "drive_folder_id", "folder1")

        file_entry = self._make_file_entry("f1", "paper.pdf", modified="2024-03-01T00:00:00.000Z")
        mock_service = MagicMock()
        mock_service.files().list().execute.return_value = {
            "files": [file_entry],
            "nextPageToken": None,
        }

        with (
            patch("src.drive.sync._build_drive_service", return_value=mock_service),
            patch("src.drive.sync._download_file", return_value=tmp_path / "f1.pdf"),
            patch("src.drive.sync._extract", return_value=[{"page_number": 1, "text": "Content."}]),
            patch("src.drive.sync.chunk_pages", return_value=[
                {"chunk_id": "f1_1_0", "file_id": "f1", "file_name": "paper.pdf",
                 "page_number": 1, "chunk_index": 0, "text": "Content."}
            ]),
            patch("src.drive.sync.upsert_chunks", return_value=["f1_1_0"]),
            patch("src.drive.sync.delete_by_file_id"),
        ):
            from src.drive.sync import sync_drive
            sync_drive()

        # Assert — state file written with correct data
        state = json.loads(state_path.read_text())
        assert "f1" in state
        assert state["f1"]["modified_time"] == "2024-03-01T00:00:00.000Z"
        assert state["f1"]["chunk_ids"] == ["f1_1_0"]


class TestLoadSyncState:
    def test_returns_empty_dict_when_file_missing(self, tmp_path, monkeypatch):
        # Arrange
        monkeypatch.setattr(sync_module.settings, "sync_state_path", tmp_path / "missing.json")

        # Act
        state = _load_sync_state()

        # Assert
        assert state == {}

    def test_loads_existing_state(self, tmp_path, monkeypatch):
        # Arrange
        state_path = tmp_path / "state.json"
        state_path.write_text(json.dumps({"f1": {"name": "paper.pdf"}}))
        monkeypatch.setattr(sync_module.settings, "sync_state_path", state_path)

        # Act
        state = _load_sync_state()

        # Assert
        assert state == {"f1": {"name": "paper.pdf"}}


class TestSaveSyncState:
    def test_writes_state_to_file(self, tmp_path, monkeypatch):
        # Arrange
        state_path = tmp_path / "state.json"
        monkeypatch.setattr(sync_module.settings, "sync_state_path", state_path)

        # Act
        _save_sync_state({"f1": {"name": "paper.pdf"}})

        # Assert
        assert state_path.exists()
        assert json.loads(state_path.read_text()) == {"f1": {"name": "paper.pdf"}}
