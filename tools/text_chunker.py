from __future__ import annotations

from typing import Any

from .base import BaseTool


class TextChunkerTool(BaseTool):
    """Chunk large text into fixed-size segments."""

    name = "text_chunker"
    description = "Split text into chunks for stepwise analysis."

    def run(self, context: dict[str, Any]) -> str:
        text = str(context.get("text") or context.get("file_content") or "")
        if not text:
            return "text_chunker: missing 'text' or 'file_content'."

        chunk_size = max(200, int(context.get("chunk_size", 1200)))
        overlap = max(0, min(int(context.get("chunk_overlap", 100)), chunk_size - 1))

        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]
            chunks.append(chunk)
            if end == len(text):
                break
            start = end - overlap

        lines: list[str] = [f"Total chunks: {len(chunks)}"]
        preview_chars = int(context.get("chunk_preview_chars", 250))
        for idx, chunk in enumerate(chunks, start=1):
            preview = " ".join(chunk.split())[:preview_chars]
            lines.append(f"[{idx}] {preview}")

        return "\n".join(lines)
