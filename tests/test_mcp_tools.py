import json
from unittest.mock import patch

import pytest

from src.mcp_server.server import call_tool, list_tools


class TestListTools:
    @pytest.mark.asyncio
    async def test_returns_three_tools(self):
        # Act
        tools = await list_tools()

        # Assert
        assert len(tools) == 3
        names = {t.name for t in tools}
        assert names == {"search_references", "list_papers", "sync_drive"}

    @pytest.mark.asyncio
    async def test_search_references_has_required_query_param(self):
        # Act
        tools = await list_tools()
        search_tool = next(t for t in tools if t.name == "search_references")

        # Assert
        assert "query" in search_tool.inputSchema["required"]


class TestCallTool:
    @pytest.mark.asyncio
    async def test_search_references_returns_text_content(self):
        # Arrange
        mock_results = [
            {"chunk_id": "f1_1_0", "file_name": "paper.pdf", "page_number": 1,
             "text": "Sample text.", "score": 0.95}
        ]

        with patch("src.mcp_server.server.query_chunks", return_value=mock_results):
            # Act
            result = await call_tool("search_references", {"query": "climate", "top_k": 3})

        # Assert
        assert len(result) == 1
        assert result[0].type == "text"
        payload = json.loads(result[0].text)
        assert payload == mock_results

    @pytest.mark.asyncio
    async def test_search_references_uses_default_top_k(self):
        # Arrange
        with patch("src.mcp_server.server.query_chunks", return_value=[]) as mock_query:
            # Act
            await call_tool("search_references", {"query": "methods"})

        # Assert
        mock_query.assert_called_once_with(query="methods", top_k=5)

    @pytest.mark.asyncio
    async def test_list_papers_returns_paper_list(self):
        # Arrange
        mock_papers = [{"file_id": "f1", "file_name": "paper.pdf", "chunk_count": 3}]

        with patch("src.mcp_server.server.list_papers", return_value=mock_papers):
            # Act
            result = await call_tool("list_papers", {})

        # Assert
        payload = json.loads(result[0].text)
        assert payload == mock_papers

    @pytest.mark.asyncio
    async def test_sync_drive_returns_summary(self):
        # Arrange
        mock_summary = {"added": 2, "updated": 0, "skipped": 1, "total": 3}

        with patch("src.mcp_server.server.sync_drive", return_value=mock_summary):
            # Act
            result = await call_tool("sync_drive", {})

        # Assert
        payload = json.loads(result[0].text)
        assert payload == mock_summary

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self):
        # Act
        result = await call_tool("nonexistent_tool", {})

        # Assert
        payload = json.loads(result[0].text)
        assert "error" in payload
        assert "nonexistent_tool" in payload["error"]
