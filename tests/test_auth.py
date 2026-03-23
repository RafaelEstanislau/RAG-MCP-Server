import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.drive.auth import get_credentials, _save_token, SCOPES


class TestGetCredentials:
    def test_returns_valid_credentials_from_token_file(self, tmp_path, monkeypatch):
        # Arrange
        monkeypatch.setattr("src.drive.auth.settings.token_path", tmp_path / "token.json")
        monkeypatch.setattr(
            "src.drive.auth.settings.client_secret_path", tmp_path / "client_secret.json"
        )

        mock_creds = MagicMock()
        mock_creds.valid = True

        with patch("src.drive.auth.Credentials.from_authorized_user_file", return_value=mock_creds):
            (tmp_path / "token.json").write_text("{}")

            # Act
            result = get_credentials()

        # Assert
        assert result is mock_creds

    def test_refreshes_expired_credentials(self, tmp_path, monkeypatch):
        # Arrange
        monkeypatch.setattr("src.drive.auth.settings.token_path", tmp_path / "token.json")
        monkeypatch.setattr(
            "src.drive.auth.settings.client_secret_path", tmp_path / "client_secret.json"
        )

        mock_creds = MagicMock()
        mock_creds.valid = False
        mock_creds.expired = True
        mock_creds.refresh_token = "refresh-token"
        mock_creds.to_json.return_value = "{}"

        with (
            patch("src.drive.auth.Credentials.from_authorized_user_file", return_value=mock_creds),
            patch("src.drive.auth.Request"),
        ):
            (tmp_path / "token.json").write_text("{}")

            # Act
            result = get_credentials()

        # Assert
        mock_creds.refresh.assert_called_once()
        assert result is mock_creds

    def test_runs_browser_flow_when_no_token_file(self, tmp_path, monkeypatch):
        # Arrange
        monkeypatch.setattr("src.drive.auth.settings.token_path", tmp_path / "token.json")
        monkeypatch.setattr(
            "src.drive.auth.settings.client_secret_path", tmp_path / "client_secret.json"
        )

        mock_creds = MagicMock()
        mock_creds.to_json.return_value = "{}"

        mock_flow = MagicMock()
        mock_flow.run_local_server.return_value = mock_creds

        with patch(
            "src.drive.auth.InstalledAppFlow.from_client_secrets_file", return_value=mock_flow
        ):
            # Act
            result = get_credentials()

        # Assert
        mock_flow.run_local_server.assert_called_once_with(port=0)
        assert result is mock_creds

    def test_token_saved_after_browser_flow(self, tmp_path, monkeypatch):
        # Arrange
        monkeypatch.setattr("src.drive.auth.settings.token_path", tmp_path / "token.json")
        monkeypatch.setattr(
            "src.drive.auth.settings.client_secret_path", tmp_path / "client_secret.json"
        )

        mock_creds = MagicMock()
        mock_creds.to_json.return_value = '{"token": "abc"}'

        mock_flow = MagicMock()
        mock_flow.run_local_server.return_value = mock_creds

        with patch(
            "src.drive.auth.InstalledAppFlow.from_client_secrets_file", return_value=mock_flow
        ):
            get_credentials()

        # Assert
        assert (tmp_path / "token.json").exists()
        assert json.loads((tmp_path / "token.json").read_text())["token"] == "abc"


class TestSaveToken:
    def test_creates_parent_directory_if_missing(self, tmp_path):
        # Arrange
        token_path = tmp_path / "nested" / "dir" / "token.json"
        mock_creds = MagicMock()
        mock_creds.to_json.return_value = "{}"

        with patch("src.drive.auth.settings.token_path", token_path):
            # Act
            _save_token(mock_creds)

        # Assert
        assert token_path.exists()
