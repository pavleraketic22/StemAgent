from __future__ import annotations

import re
from typing import Any

from .base import BaseTool


class CitationExtractorTool(BaseTool):
    """Extract URLs and DOI-like references from text."""

    name = "citation_extractor"
    description = "Extract citation candidates (URLs/DOIs) from text."

    _URL_PATTERN = re.compile(r"https?://[^\s)\]>'\"]+", re.IGNORECASE)
    _DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)

    def run(self, context: dict[str, Any]) -> str:
        text = str(context.get("text") or context.get("answer") or "")
        if not text:
            return "citation_extractor: missing 'text' or 'answer'."

        urls = sorted(set(self._URL_PATTERN.findall(text)))
        dois = sorted(set(self._DOI_PATTERN.findall(text)))

        lines: list[str] = [f"urls={len(urls)}", f"dois={len(dois)}"]
        lines.extend(f"URL: {u}" for u in urls[:50])
        lines.extend(f"DOI: {d}" for d in dois[:50])
        return "\n".join(lines)
