import io
import json
from pathlib import Path
from typing import Any

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from config.settings import settings
from src.drive.auth import get_credentials
from src.ingestion.extractor import extract_text_pdf, extract_text_docx
from src.ingestion.chunker import chunk_pages
from src.store.vector_store import upsert_chunks, delete_by_file_id


SUPPORTED_MIME_TYPES = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
}


def sync_drive() -> dict[str, Any]:
    """Sync Drive folder to the local vector store. Returns a summary dict."""
    service = _build_drive_service()
    state = _load_sync_state()
    drive_files = _list_drive_files(service)

    added, updated, skipped = 0, 0, 0

    for file in drive_files:
        file_id = file["id"]
        modified_time = file["modifiedTime"]
        file_name = file["name"]
        mime_type = file["mimeType"]

        stored = state.get(file_id)
        if stored and stored["modified_time"] == modified_time:
            skipped += 1
            continue

        if stored:
            delete_by_file_id(file_id)
            updated += 1
        else:
            added += 1

        local_path = _download_file(service, file_id, file_name, mime_type)
        pages = _extract(local_path, mime_type)
        chunks = chunk_pages(
            pages,
            file_id=file_id,
            file_name=file_name,
            max_tokens=settings.chunk_max_tokens,
            overlap_paragraphs=settings.chunk_overlap_paragraphs,
        )
        chunk_ids = upsert_chunks(chunks)

        state[file_id] = {
            "name": file_name,
            "modified_time": modified_time,
            "chunk_ids": chunk_ids,
        }
        _save_sync_state(state)

    return {"added": added, "updated": updated, "skipped": skipped, "total": len(drive_files)}


def _build_drive_service():
    creds = get_credentials()
    return build("drive", "v3", credentials=creds)


def _list_drive_files(service) -> list[dict]:
    """List all supported files in the configured Drive folder (non-recursive for simplicity)."""
    mime_filter = " or ".join(
        f"mimeType='{m}'" for m in SUPPORTED_MIME_TYPES
    )
    query = f"'{settings.drive_folder_id}' in parents and ({mime_filter}) and trashed=false"

    results = []
    page_token = None
    while True:
        response = (
            service.files()
            .list(
                q=query,
                fields="nextPageToken, files(id, name, mimeType, modifiedTime)",
                pageToken=page_token,
            )
            .execute()
        )
        results.extend(response.get("files", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            break

    return results


def _download_file(service, file_id: str, file_name: str, mime_type: str) -> Path:
    ext = SUPPORTED_MIME_TYPES[mime_type]
    settings.downloads_path.mkdir(parents=True, exist_ok=True)
    local_path = settings.downloads_path / f"{file_id}{ext}"

    request = service.files().get_media(fileId=file_id)
    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()

    local_path.write_bytes(buffer.getvalue())
    return local_path


def _extract(local_path: Path, mime_type: str) -> list[dict]:
    if mime_type == "application/pdf":
        return extract_text_pdf(local_path)
    return extract_text_docx(local_path)


def _load_sync_state() -> dict:
    if settings.sync_state_path.exists():
        return json.loads(settings.sync_state_path.read_text())
    return {}


def _save_sync_state(state: dict) -> None:
    settings.sync_state_path.parent.mkdir(parents=True, exist_ok=True)
    settings.sync_state_path.write_text(json.dumps(state, indent=2))
