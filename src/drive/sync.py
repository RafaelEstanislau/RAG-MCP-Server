import gc
import io
import logging
from typing import Any

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from config.settings import settings
from src.drive.auth import get_credentials
from src.ingestion.extractor import extract_text_pdf, extract_text_docx
from src.ingestion.chunker import chunk_pages
from src.store.vector_store import upsert_chunks, delete_by_file_id, list_indexed_files

logger = logging.getLogger(__name__)

SUPPORTED_MIME_TYPES = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
}


def sync_drive() -> dict[str, Any]:
    """Sync Drive folder to the vector store. Returns a summary dict."""
    logger.info("sync_drive: starting")
    service = _build_drive_service()
    indexed = list_indexed_files()
    drive_files = _list_drive_files(service)
    logger.info("sync_drive: found %d drive files, %d already indexed", len(drive_files), len(indexed))

    added, updated, skipped, removed = 0, 0, 0, 0
    drive_file_ids = {f["id"] for f in drive_files}

    # Remove files that were deleted from Drive
    for file_id in indexed:
        if file_id not in drive_file_ids:
            delete_by_file_id(file_id)
            removed += 1

    for file in drive_files:
        file_id = file["id"]
        modified_time = file["modifiedTime"]
        file_name = file["name"]
        mime_type = file["mimeType"]

        if file_id in indexed and indexed[file_id] == modified_time:
            skipped += 1
            continue

        if file_id in indexed:
            delete_by_file_id(file_id)
            updated += 1
        else:
            added += 1

        logger.info("sync_drive: processing '%s'", file_name)
        data = _download_file(service, file_id, mime_type)
        pages = _extract(data, mime_type)
        del data
        gc.collect()

        # Process one page at a time to keep peak memory low on free-tier hosting.
        # chunk_pages already resets per page (no cross-page state), so streaming is safe.
        logger.info("sync_drive: '%s' — %d pages, embedding page by page", file_name, len(pages))
        for page in pages:
            chunks = chunk_pages(
                [page],
                file_id=file_id,
                file_name=file_name,
                max_tokens=settings.chunk_max_tokens,
                overlap_paragraphs=settings.chunk_overlap_paragraphs,
            )
            for chunk in chunks:
                chunk["modified_time"] = modified_time
            upsert_chunks(chunks)
            del chunks
        del pages
        gc.collect()
        logger.info("sync_drive: '%s' done", file_name)

    logger.info("sync_drive: complete — added=%d updated=%d skipped=%d removed=%d", added, updated, skipped, removed)
    return {"added": added, "updated": updated, "skipped": skipped, "removed": removed, "total": len(drive_files)}


def _build_drive_service():
    creds = get_credentials()
    # cache_discovery=False skips fetching/caching the API discovery document,
    # which can spike 50-100 MB on a cold start on memory-constrained hosts.
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _list_drive_files(service) -> list[dict]:
    """Recursively list all supported files under the configured Drive folder (BFS)."""
    mime_filter = " or ".join(f"mimeType='{m}'" for m in SUPPORTED_MIME_TYPES)
    folder_queue = [settings.drive_folder_id]
    visited = set()
    all_files = []

    while folder_queue:
        folder_id = folder_queue.pop(0)
        if folder_id in visited:
            continue
        visited.add(folder_id)

        # Collect subfolders
        page_token = None
        while True:
            response = (
                service.files()
                .list(
                    q=f"'{folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false",
                    fields="nextPageToken, files(id)",
                    pageToken=page_token,
                )
                .execute()
            )
            for f in response.get("files", []):
                folder_queue.append(f["id"])
            page_token = response.get("nextPageToken")
            if not page_token:
                break

        # Collect supported files in this folder
        page_token = None
        while True:
            response = (
                service.files()
                .list(
                    q=f"'{folder_id}' in parents and ({mime_filter}) and trashed=false",
                    fields="nextPageToken, files(id, name, mimeType, modifiedTime)",
                    pageToken=page_token,
                )
                .execute()
            )
            all_files.extend(response.get("files", []))
            page_token = response.get("nextPageToken")
            if not page_token:
                break

    return all_files


def _download_file(service, file_id: str, mime_type: str) -> bytes:
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, service.files().get_media(fileId=file_id))
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buf.getvalue()


def _extract(data: bytes, mime_type: str) -> list[dict]:
    if mime_type == "application/pdf":
        return extract_text_pdf(data)
    return extract_text_docx(data)
