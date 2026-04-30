from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import BaseTool


class FileReadTool(BaseTool):
    """Read local files in a safe, bounded way."""

    name = "file_read"
    description = "Read a local file (bounded by max chars)."

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = (base_dir or Path.cwd()).resolve()

    def run(self, context: dict[str, Any]) -> str:
        raw_path = str(context.get("file_path") or "").strip()
        if not raw_path:
            return "file_read: missing 'file_path'."

        max_chars = int(context.get("file_max_chars", 8000))
        try:
            safe_path = self._resolve_safe_path(raw_path)
            if not safe_path.exists():
                return f"file_read: file does not exist: {safe_path}"
            if safe_path.is_dir():
                entries = sorted(p.name + ("/" if p.is_dir() else "") for p in safe_path.iterdir())
                return "\n".join(entries)

            content = safe_path.read_text(encoding="utf-8")
            return content[:max_chars]
        except Exception as exc:
            return f"file_read error: {exc}"

    def _resolve_safe_path(self, raw_path: str) -> Path:
        path = Path(raw_path)
        resolved = (path if path.is_absolute() else (self.base_dir / path)).resolve()
        if self.base_dir not in resolved.parents and resolved != self.base_dir:
            raise ValueError("path is outside allowed base directory")
        return resolved
