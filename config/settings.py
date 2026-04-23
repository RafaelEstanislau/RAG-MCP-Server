from dataclasses import dataclass, field
from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent


@dataclass
class Settings:
    drive_folder_id: str = field(default_factory=lambda: os.getenv("DRIVE_FOLDER_ID", ""))
    qdrant_url: str = field(default_factory=lambda: os.getenv("QDRANT_URL", ""))
    qdrant_api_key: str = field(default_factory=lambda: os.getenv("QDRANT_API_KEY", ""))
    google_service_account_key: str = field(
        default_factory=lambda: os.getenv("GOOGLE_SERVICE_ACCOUNT_KEY", "")
    )
    embed_model: str = field(default_factory=lambda: os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
    chunk_max_tokens: int = field(default_factory=lambda: int(os.getenv("CHUNK_MAX_TOKENS", "400")))
    chunk_overlap_paragraphs: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP_PARAGRAPHS", "1"))
    )
    mcp_host: str = field(default_factory=lambda: os.getenv("MCP_HOST", "0.0.0.0"))
    mcp_port: int = field(default_factory=lambda: int(os.getenv("MCP_PORT", os.getenv("PORT", "8000"))))
    mcp_server_url: str = field(default_factory=lambda: os.getenv("MCP_SERVER_URL", "http://localhost:8000"))


settings = Settings()
