# RAG MCP Server

A personal semantic search engine over your Google Drive reference library, exposed as an MCP server and connectable to Claude.ai.

---

## Why this instead of Claude.ai's native Google Drive integration?

| | Native Drive connection | This RAG server |
|---|---|---|
| **How it works** | Dumps raw file content into context | Embeds chunks as vectors, retrieves only relevant ones |
| **Large libraries** | Hits token limits fast with 10+ papers | Scales to hundreds of documents — only top-K chunks are returned |
| **Search quality** | Keyword/filename match | Semantic similarity — finds relevant content even without exact keywords |
| **PDF handling** | Full text dump | Per-page extraction with page number metadata |
| **Re-reads files** | Every conversation | Index is persisted in ChromaDB — no re-download needed |
| **Citation support** | No | Returns paper title + page number with every chunk |
| **Context efficiency** | Entire documents in context | Only the most relevant passages, keeping the window clean |

The core idea: instead of giving Claude the whole library, you give it a tool to *query* the library. Claude decides what to search for and gets back focused, cited excerpts.

---

## Architecture

```
Google Drive (PDFs / DOCX)
        │
        ▼
   src/drive/          ← OAuth 2.0 auth + file sync
        │
        ▼
   src/ingestion/      ← text extraction (PyMuPDF / python-docx) + chunking
        │
        ▼
   src/store/          ← sentence-transformers embeddings → ChromaDB
        │
        ▼
   src/mcp_server/     ← MCP tools over Streamable HTTP + OAuth 2.0
        │
        ▼
     Claude.ai
```

---

## Prerequisites

- Python 3.11+
- A Google Cloud project with the Drive API enabled
- A Google Drive folder containing your PDF/DOCX papers
- A public HTTPS URL for your server (ngrok works for local dev)

---

## Step-by-step setup

### 1. Clone and install

```bash
git clone <repo-url>
cd RAG
pip install -e ".[dev]"
```

### 2. Google Cloud — create credentials

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project (or select an existing one)
3. Enable the **Google Drive API**: APIs & Services → Enable APIs → search "Drive API"
4. Create OAuth credentials: APIs & Services → Credentials → Create Credentials → **OAuth client ID**
   - Application type: **Desktop app**
   - Download the JSON file
5. Save it as `credentials/client_secret.json` in this project root

### 3. Get your Drive folder ID

Open the Google Drive folder you want to index in your browser. The URL looks like:

```
https://drive.google.com/drive/folders/1A2B3C4D5E6F7G8H9I0J
```

The long string at the end (`1A2B3C4D5E6F7G8H9I0J`) is your folder ID.

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
DRIVE_FOLDER_ID=your_folder_id_here
MCP_SERVER_URL=https://your-public-url.example.com
```

`MCP_SERVER_URL` must be the public HTTPS URL that Claude.ai will use to reach your server (see step 6).

Optional variables (with defaults):

```env
MCP_HOST=0.0.0.0
MCP_PORT=8000
EMBED_MODEL=all-MiniLM-L6-v2
CHUNK_MAX_TOKENS=400
CHUNK_OVERLAP_PARAGRAPHS=1
```

### 5. First run — authorize Google Drive access

Run the server once to trigger the Drive OAuth flow:

```bash
python -m src.mcp_server.server
```

A browser window will open asking you to authorize access to your Drive. After approving, the token is saved to `credentials/token.json` and reused on future runs.

### 6. Expose the server publicly (ngrok)

Claude.ai is a remote client — it needs to reach your server over the internet via HTTPS.

```bash
# in a separate terminal
ngrok http 8000
```

Copy the `https://` URL ngrok gives you (e.g. `https://abc123.ngrok-free.app`) and update your `.env`:

```env
MCP_SERVER_URL=https://abc123.ngrok-free.app
```

Then restart the server:

```bash
python -m src.mcp_server.server
```

> **Note:** Free ngrok URLs change on every restart. Update `MCP_SERVER_URL` and restart the server each time. A paid ngrok plan gives you a stable domain.

### 7. Connect Claude.ai

1. Go to [claude.ai](https://claude.ai) → Settings → Integrations (or the MCP section)
2. Add a new MCP server with the URL from step 6
3. Claude.ai will open a browser popup to complete the OAuth flow — approve it
4. The server is now connected

### 8. Initial sync

In a Claude.ai conversation:

> "Use the sync_drive tool to download and index my papers"

This downloads all PDFs and DOCX files from your Drive folder and builds the vector index. Subsequent syncs only process new or changed files.

---

## Available tools

| Tool | Description |
|---|---|
| `search_references` | Semantic search — returns the most relevant chunks with paper title and page number |
| `list_papers` | Lists all papers currently indexed |
| `sync_drive` | Downloads new/updated files from Drive and re-indexes them |

### Example prompts

```
Search my references for transformer attention mechanisms
```
```
List all papers I have indexed
```
```
Sync my Drive folder, then find papers about retrieval-augmented generation and summarize the key ideas
```

---

## Project structure

```
RAG/
├── config/
│   └── settings.py          # All configuration via env vars
├── credentials/
│   ├── client_secret.json   # Google OAuth app credentials (git-ignored)
│   └── token.json           # Drive access token (git-ignored)
├── data/
│   ├── chroma/              # Vector store (git-ignored)
│   ├── downloads/           # Downloaded papers (git-ignored)
│   └── sync_state.json      # Tracks synced files (git-ignored)
├── src/
│   ├── drive/
│   │   ├── auth.py          # Google Drive OAuth 2.0
│   │   └── sync.py          # File download and sync logic
│   ├── ingestion/
│   │   ├── extractor.py     # PDF (per-page) and DOCX text extraction
│   │   └── chunker.py       # Token-aware chunking with overlap
│   ├── store/
│   │   └── vector_store.py  # Embedding + ChromaDB storage and query
│   └── mcp_server/
│       ├── oauth.py         # In-memory OAuth 2.0 provider (auto-approves)
│       └── server.py        # MCP tools + Streamable HTTP transport
├── tests/                   # Full test suite (AAA pattern, 90%+ coverage)
├── .env                     # Local env vars (git-ignored)
└── pyproject.toml
```

---

## Running tests

```bash
pytest
```
