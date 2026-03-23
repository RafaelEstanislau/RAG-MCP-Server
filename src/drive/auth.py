from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from config.settings import settings


SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def get_credentials() -> Credentials:
    """Return valid Drive API credentials, triggering OAuth browser flow on first run."""
    if settings.token_path.exists():
        creds = Credentials.from_authorized_user_file(str(settings.token_path), SCOPES)
        if creds.valid:
            return creds
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            _save_token(creds)
            return creds

    creds = _run_browser_flow()
    _save_token(creds)
    return creds


def _run_browser_flow() -> Credentials:
    flow = InstalledAppFlow.from_client_secrets_file(
        str(settings.client_secret_path), SCOPES
    )
    return flow.run_local_server(port=0)


def _save_token(creds: Credentials) -> None:
    settings.token_path.parent.mkdir(parents=True, exist_ok=True)
    settings.token_path.write_text(creds.to_json())


if __name__ == "__main__":
    get_credentials()
    print("Authentication successful. Token saved.")
