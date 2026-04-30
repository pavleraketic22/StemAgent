from __future__ import annotations

from pathlib import Path
from typing import Any

from .arxiv_search import ArxivSearchTool
from .base import BaseTool
from .checklist_validator import ChecklistValidatorTool
from .citation_extractor import CitationExtractorTool
from .diff_compare import DiffCompareTool
from .file_read import FileReadTool
from .file_write import FileWriteTool
from .security_pattern_scan import SecurityPatternScanTool
from .text_chunker import TextChunkerTool
from .url_fetch import UrlFetchTool
from .web_search import WebSearchTool
from .wikipedia_search import WikipediaSearchTool


class ToolLibrary:
    """Mutable runtime registry for tools.

    Enables runtime extension by evolved agents.
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def as_dict(self) -> dict[str, BaseTool]:
        return dict(self._tools)

    def list_tools(self) -> list[dict[str, str]]:
        return [
            {"name": name, "description": tool.description}
            for name, tool in sorted(self._tools.items())
        ]


TOOL_LIBRARY = ToolLibrary()


def _register_default_tools() -> None:
    base_dir = Path.cwd()
    TOOL_LIBRARY.register(WebSearchTool())
    TOOL_LIBRARY.register(UrlFetchTool())
    TOOL_LIBRARY.register(ArxivSearchTool())
    TOOL_LIBRARY.register(WikipediaSearchTool())
    TOOL_LIBRARY.register(CitationExtractorTool())
    TOOL_LIBRARY.register(TextChunkerTool())
    TOOL_LIBRARY.register(DiffCompareTool())
    TOOL_LIBRARY.register(ChecklistValidatorTool())
    TOOL_LIBRARY.register(SecurityPatternScanTool())
    TOOL_LIBRARY.register(FileReadTool(base_dir=base_dir))
    TOOL_LIBRARY.register(FileWriteTool(base_dir=base_dir))


_register_default_tools()

# Backward-compatible export used by Agent.
AVAILABLE_TOOLS: dict[str, BaseTool] = TOOL_LIBRARY.as_dict()


def register_tool(tool: BaseTool) -> None:
    """Register new tool at runtime and refresh exported mapping."""
    TOOL_LIBRARY.register(tool)
    AVAILABLE_TOOLS.clear()
    AVAILABLE_TOOLS.update(TOOL_LIBRARY.as_dict())


def unregister_tool(name: str) -> None:
    """Unregister tool at runtime and refresh exported mapping."""
    TOOL_LIBRARY.unregister(name)
    AVAILABLE_TOOLS.clear()
    AVAILABLE_TOOLS.update(TOOL_LIBRARY.as_dict())
