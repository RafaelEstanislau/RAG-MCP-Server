from dataclasses import dataclass, field
from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent


@dataclass
class Settings:
    drive_folder_id: str = field(default_factory=lambda: os.getenv("DRIVE_FOLDER_ID", ""))
    chroma_path: Path = field(default_factory=lambda: BASE_DIR / "data" / "chroma")
    downloads_path: Path = field(default_factory=lambda: BASE_DIR / "data" / "downloads")
    sync_state_path: Path = field(default_factory=lambda: BASE_DIR / "data" / "sync_state.json")
    client_secret_path: Path = field(default_factory=lambda: BASE_DIR / "credentials" / "client_secret.json")
    token_path: Path = field(default_factory=lambda: BASE_DIR / "credentials" / "token.json")
    embed_model: str = field(default_factory=lambda: os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2"))
    chunk_max_tokens: int = field(default_factory=lambda: int(os.getenv("CHUNK_MAX_TOKENS", "400")))
    chunk_overlap_paragraphs: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP_PARAGRAPHS", "1"))
    )
    mcp_host: str = field(default_factory=lambda: os.getenv("MCP_HOST", "0.0.0.0"))
    mcp_port: int = field(default_factory=lambda: int(os.getenv("MCP_PORT", "8000")))
    mcp_server_url: str = field(default_factory=lambda: os.getenv("MCP_SERVER_URL", "http://localhost:8000"))


settings = Settings()
