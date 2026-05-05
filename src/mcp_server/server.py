import contextlib
import json
import logging

import uvicorn

logger = logging.getLogger(__name__)
from mcp.server import Server
from mcp.server.auth.middleware.bearer_auth import BearerAuthBackend, RequireAuthMiddleware
from mcp.server.auth.provider import ProviderTokenVerifier
from mcp.server.auth.routes import create_auth_routes, create_protected_resource_routes
from mcp.server.auth.settings import ClientRegistrationOptions
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.types import TextContent, Tool
from pydantic import AnyHttpUrl
from starlette.applications import Starlette
from starlette.middleware.authentication import AuthenticationMiddleware
from starlette.routing import Mount, Route

from config.settings import settings
from src.drive.sync import sync_drive
from src.mcp_server.oauth import SimpleOAuthProvider
from src.store.vector_store import list_papers, query_chunks, warmup


app = Server("rag-reference-server")


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_references",
            description=(
                "Semantically search the indexed reference papers. "
                "Returns the most relevant text chunks with paper title and page number."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query describing what you are looking for.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of chunks to return (default 5).",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="list_papers",
            description="List all papers currently indexed in the reference store.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="sync_drive",
            description=(
                "Sync the Google Drive folder with the local index. "
                "Downloads new or updated papers and re-indexes them."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        if name == "search_references":
            query = arguments.get("query", "")
            top_k = int(arguments.get("top_k", 5))
            results = query_chunks(query=query, top_k=top_k)
            return [TextContent(type="text", text=json.dumps(results, ensure_ascii=False, indent=2))]

        if name == "list_papers":
            papers = list_papers()
            return [TextContent(type="text", text=json.dumps(papers, ensure_ascii=False, indent=2))]

        if name == "sync_drive":
            summary = sync_drive()
            return [TextContent(type="text", text=json.dumps(summary, ensure_ascii=False, indent=2))]

        return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]

    except Exception as exc:
        logger.exception("Tool %s failed", name)
        return [TextContent(type="text", text=json.dumps({"error": str(exc)}))]


def build_starlette_app(server_url: str) -> Starlette:
    oauth_provider = SimpleOAuthProvider()
    issuer_url = AnyHttpUrl(server_url)
    resource_url = AnyHttpUrl(server_url)
    token_verifier = ProviderTokenVerifier(oauth_provider)

    session_manager = StreamableHTTPSessionManager(app=app, stateless=False)

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        warmup()
        async with session_manager.run():
            yield

    # Wrap MCP handler with bearer auth — only this endpoint requires a valid token.
    # AuthenticationMiddleware populates scope["user"]; RequireAuthMiddleware enforces it.
    # Mount (not Route) is used so the raw ASGI callable receives (scope, receive, send)
    # directly, instead of being wrapped by Starlette's request_response() adapter.
    protected_mcp = AuthenticationMiddleware(
        app=RequireAuthMiddleware(
            app=session_manager.handle_request,
            required_scopes=[],
        ),
        backend=BearerAuthBackend(token_verifier),
    )

    auth_routes = create_auth_routes(
        provider=oauth_provider,
        issuer_url=issuer_url,
        client_registration_options=ClientRegistrationOptions(enabled=True),
    )
    resource_routes = create_protected_resource_routes(
        resource_url=resource_url,
        authorization_servers=[issuer_url],
    )

    return Starlette(
        # auth_routes and resource_routes are matched first (exact paths).
        # Mount("/") catches everything else — GET /, POST /, DELETE / for MCP.
        routes=auth_routes + resource_routes + [Mount("/", app=protected_mcp)],
        lifespan=lifespan,
    )


if __name__ == "__main__":  # pragma: no cover
    starlette_app = build_starlette_app(settings.mcp_server_url)
    uvicorn.run(starlette_app, host=settings.mcp_host, port=settings.mcp_port)
