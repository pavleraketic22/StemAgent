from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any

from .base import BaseTool


class WikipediaSearchTool(BaseTool):
    name = "wikipedia_search"
    description = "Search Wikipedia and fetch short article extracts."

    def run(self, context: dict[str, Any]) -> str:
        query = str(context.get("wikipedia_query") or context.get("question") or "").strip()
        if not query:
            return "wikipedia_search: missing query (expected 'wikipedia_query' or 'question')."

        limit = int(context.get("wikipedia_limit", 5))
        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "format": "json",
            "utf8": 1,
        }
        search_url = (
            "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(search_params)
        )

        try:
            with urllib.request.urlopen(search_url, timeout=20) as response:
                search_data = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            return f"wikipedia_search error: {exc}"

        results = search_data.get("query", {}).get("search", [])
        if not results:
            return "wikipedia_search: no results found."

        lines: list[str] = []
        for idx, item in enumerate(results[:limit], start=1):
            title = item.get("title", "")
            page_id = item.get("pageid")
            extract = self._fetch_extract(page_id)
            url_title = urllib.parse.quote(title.replace(" ", "_"))
            page_url = f"https://en.wikipedia.org/wiki/{url_title}"
            lines.append(f"[{idx}] {title}\nURL: {page_url}\nSummary: {extract}")

        return "\n\n".join(lines)

    @staticmethod
    def _fetch_extract(page_id: int | None) -> str:
        if page_id is None:
            return ""

        params = {
            "action": "query",
            "prop": "extracts",
            "pageids": page_id,
            "exintro": 1,
            "explaintext": 1,
            "format": "json",
            "utf8": 1,
        }
        url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
        try:
            with urllib.request.urlopen(url, timeout=20) as response:
                data = json.loads(response.read().decode("utf-8"))
            page = data.get("query", {}).get("pages", {}).get(str(page_id), {})
            extract = str(page.get("extract", ""))
            return " ".join(extract.split())[:500]
        except Exception:
            return ""
