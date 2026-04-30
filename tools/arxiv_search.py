from __future__ import annotations

import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any

from .base import BaseTool


class ArxivSearchTool(BaseTool):
    name = "arxiv_search"
    description = "Search arXiv and return top papers with summaries."

    def run(self, context: dict[str, Any]) -> str:
        query = str(context.get("arxiv_query") or context.get("question") or "").strip()
        if not query:
            return "arxiv_search: missing query (expected 'arxiv_query' or 'question')."

        max_results = int(context.get("arxiv_max_results", 5))
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        url = f"http://export.arxiv.org/api/query?{urllib.parse.urlencode(params)}"

        try:
            with urllib.request.urlopen(url, timeout=20) as response:
                xml_data = response.read().decode("utf-8")
        except Exception as exc:
            return f"arxiv_search error: {exc}"

        try:
            root = ET.fromstring(xml_data)
        except Exception as exc:
            return f"arxiv_search parse error: {exc}"

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall("atom:entry", ns)
        if not entries:
            return "arxiv_search: no results found."

        lines: list[str] = []
        for idx, entry in enumerate(entries[:max_results], start=1):
            title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
            summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
            link = ""
            for link_node in entry.findall("atom:link", ns):
                if link_node.attrib.get("rel") == "alternate":
                    link = link_node.attrib.get("href", "")
                    break

            summary_short = " ".join(summary.split())[:500]
            lines.append(
                f"[{idx}] {title}\nURL: {link}\nSummary: {summary_short}"
            )

        return "\n\n".join(lines)
