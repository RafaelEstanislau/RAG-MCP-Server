import json
import os

from google.oauth2 import service_account


SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def get_credentials() -> service_account.Credentials:
    key_json = os.environ["GOOGLE_SERVICE_ACCOUNT_KEY"]
    info = json.loads(key_json)
    return service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
