from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from typing import Any

from .base import BaseTool


class WebSearchTool(BaseTool):
    """Web search using SerpAPI Google engine."""

    name = "web_search"
    description = "Search the web via SerpAPI and return snippets."

    def run(self, context: dict[str, Any]) -> str:
        query = str(context.get("search_query") or context.get("question") or "").strip()
        if not query:
            return "web_search: missing query (expected 'search_query' or 'question')."

        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            return "web_search: SERPAPI_API_KEY is not set in environment."

        params = {
            "q": query,
            "engine": "google",
            "api_key": api_key,
            "num": int(context.get("search_num", 5)),
        }
        url = f"https://serpapi.com/search.json?{urllib.parse.urlencode(params)}"

        try:
            with urllib.request.urlopen(url, timeout=20) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            return f"web_search error: {exc}"

        organic = payload.get("organic_results", [])
        if not organic:
            return "web_search: no organic results found."

        lines: list[str] = []
        for idx, item in enumerate(organic[: int(params["num"])]):
            title = item.get("title", "(no title)")
            link = item.get("link", "")
            snippet = item.get("snippet", "")
            lines.append(f"[{idx + 1}] {title}\nURL: {link}\nSnippet: {snippet}")

        return "\n\n".join(lines)
