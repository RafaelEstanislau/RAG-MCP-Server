import json
from unittest.mock import MagicMock, patch

import pytest

from src.drive.auth import get_credentials, SCOPES


class TestGetCredentials:
    def test_returns_service_account_credentials(self, monkeypatch):
        # Arrange
        key_data = {
            "type": "service_account",
            "project_id": "test-project",
            "private_key_id": "key-id",
            "client_email": "test@test-project.iam.gserviceaccount.com",
            "client_id": "123",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
        monkeypatch.setenv("GOOGLE_SERVICE_ACCOUNT_KEY", json.dumps(key_data))

        mock_creds = MagicMock()
        with patch(
            "src.drive.auth.service_account.Credentials.from_service_account_info",
            return_value=mock_creds,
        ) as mock_factory:
            # Act
            result = get_credentials()

        # Assert
        mock_factory.assert_called_once_with(key_data, scopes=SCOPES)
        assert result is mock_creds

    def test_raises_when_env_var_missing(self, monkeypatch):
        # Arrange
        monkeypatch.delenv("GOOGLE_SERVICE_ACCOUNT_KEY", raising=False)

        # Act & Assert
        with pytest.raises(KeyError):
            get_credentials()

    def test_raises_on_invalid_json(self, monkeypatch):
        # Arrange
        monkeypatch.setenv("GOOGLE_SERVICE_ACCOUNT_KEY", "not-valid-json")

        # Act & Assert
        with pytest.raises(json.JSONDecodeError):
            get_credentials()
