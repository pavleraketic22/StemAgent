from __future__ import annotations

import json
import urllib.request
from typing import Any

from .base import BaseTool


class UrlFetchTool(BaseTool):
    """Fetch raw text/HTML content from a URL."""

    name = "url_fetch"
    description = "Fetch content from a URL for deeper analysis."

    def run(self, context: dict[str, Any]) -> str:
        url = str(context.get("url") or "").strip()
        if not url:
            return "url_fetch: missing 'url'."

        max_chars = int(context.get("url_max_chars", 12000))
        timeout = int(context.get("url_timeout", 20))

        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/123.0.0.0 Safari/537.36"
                )
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
                content_type = response.headers.get("Content-Type", "")
        except Exception as exc:
            return f"url_fetch error: {exc}"

        if "application/json" in content_type:
            try:
                data = json.loads(raw.decode("utf-8", errors="replace"))
                text = json.dumps(data, ensure_ascii=False, indent=2)
            except Exception:
                text = raw.decode("utf-8", errors="replace")
        else:
            text = raw.decode("utf-8", errors="replace")

        return text[:max_chars]
