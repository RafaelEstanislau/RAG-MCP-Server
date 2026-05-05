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
| **Re-reads files** | Every conversation | Index is persisted in Qdrant Cloud — no re-download needed |
| **Citation support** | No | Returns paper title + page number with every chunk |
| **Context efficiency** | Entire documents in context | Only the most relevant passages, keeping the window clean |

The core idea: instead of giving Claude the whole library, you give it a tool to *query* the library. Claude decides what to search for and gets back focused, cited excerpts.

---

## Architecture

```
Google Drive (PDFs / DOCX)
        │
        ▼
   src/drive/          ← Service Account auth + file sync
        │
        ▼
   src/ingestion/      ← text extraction (PyMuPDF / python-docx) + chunking
        │
        ▼
   src/store/          ← fastembed embeddings → Qdrant Cloud
        │
        ▼
   src/mcp_server/     ← MCP tools over Streamable HTTP + OAuth 2.0
        │
        ▼
     Claude.ai
```

---

## Prerequisites

- Python 3.11–3.13 (3.14 not supported — fastembed has no wheels for it yet)
- A Google Cloud project with the Drive API enabled and a **Service Account**
- A Google Drive folder shared with that service account
- A [Qdrant Cloud](https://cloud.qdrant.io/) cluster (free tier works)
- A public HTTPS URL for your server (Render free tier or ngrok for local dev)

---

## Step-by-step setup

### 1. Clone and install

```bash
git clone <repo-url>
cd RAG-MCP-Server
pip install -e ".[dev]"
```

### 2. Qdrant Cloud — create a cluster

1. Sign up at [cloud.qdrant.io](https://cloud.qdrant.io/)
2. Create a free cluster
3. Copy the **Cluster URL** and generate an **API key** from the dashboard

### 3. Google Cloud — create a Service Account

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create or select a project and enable the **Google Drive API**
3. Go to **IAM & Admin → Service Accounts → Create Service Account**
4. Grant it no special roles (Drive access is controlled by folder sharing)
5. Open the service account → **Keys → Add Key → JSON** — download the file
6. Copy the entire JSON file contents — you'll paste it as an env var below

### 4. Share your Drive folder with the service account

Open the Google Drive folder you want to index, click **Share**, and add the service account email (looks like `name@project.iam.gserviceaccount.com`) with **Viewer** access.

Get the folder ID from the URL:

```
https://drive.google.com/drive/folders/1A2B3C4D5E6F7G8H9I0J
                                        ^^^^^^^^^^^^^^^^^^^^
                                        this is your folder ID
```

### 5. Configure environment variables

Create a `.env` file in the project root:

```env
DRIVE_FOLDER_ID=your_folder_id_here
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
GOOGLE_SERVICE_ACCOUNT_KEY={"type":"service_account","project_id":"..."}
MCP_SERVER_URL=https://your-public-url.example.com
```

`GOOGLE_SERVICE_ACCOUNT_KEY` is the full contents of the downloaded JSON key, on a single line.

Optional variables (with defaults):

```env
MCP_HOST=0.0.0.0
MCP_PORT=8000
EMBED_MODEL=BAAI/bge-small-en-v1.5
CHUNK_MAX_TOKENS=400
CHUNK_OVERLAP_PARAGRAPHS=1
```

> **Memory tip (free-tier hosting):** If you hit the 512 MB limit during sync, lower `CHUNK_MAX_TOKENS` to `200`. Smaller chunks mean fewer vectors in memory per embedding batch.

### 6. Run the server

```bash
python -m src.mcp_server.server
```

No browser OAuth flow is needed — the service account authenticates automatically.

### 7. Expose the server publicly

**Option A — deploy to Render (recommended)**

1. Push the repo to GitHub
2. Create a new **Web Service** on [render.com](https://render.com), connected to your repo
3. Set the build command: `pip install -e .`
4. Set the start command: `python -m src.mcp_server.server`
5. Add all env vars from step 5 in the Render dashboard
6. Render provides a stable `https://your-service.onrender.com` URL — use that as `MCP_SERVER_URL`

**Option B — ngrok for local dev**

```bash
# in a separate terminal
ngrok http 8000
```

Copy the `https://` URL and set it as `MCP_SERVER_URL`, then restart the server.

> Free ngrok URLs change on every restart — update `MCP_SERVER_URL` and restart each time.

### 8. Connect Claude.ai

1. Go to [claude.ai](https://claude.ai) → Settings → Integrations
2. Add a new MCP server with your public URL
3. Complete the OAuth popup that appears
4. The server is now connected

### 9. Initial sync

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
RAG-MCP-Server/
├── config/
│   └── settings.py          # All configuration via env vars
├── src/
│   ├── drive/
│   │   ├── auth.py          # Google Service Account authentication
│   │   └── sync.py          # File download and sync logic
│   ├── ingestion/
│   │   ├── extractor.py     # PDF (per-page) and DOCX text extraction
│   │   └── chunker.py       # Token-aware chunking with overlap
│   ├── store/
│   │   └── vector_store.py  # fastembed embeddings + Qdrant storage and query
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
