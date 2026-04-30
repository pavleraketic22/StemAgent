from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import BaseTool


class FileWriteTool(BaseTool):
    """Write local files in a safe, bounded way."""

    name = "file_write"
    description = "Write content to a local file (within base dir)."

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = (base_dir or Path.cwd()).resolve()

    def run(self, context: dict[str, Any]) -> str:
        raw_path = str(context.get("file_path") or "").strip()
        content = str(context.get("file_content") or "")
        append_mode = bool(context.get("file_append", False))

        if not raw_path:
            return "file_write: missing 'file_path'."

        try:
            safe_path = self._resolve_safe_path(raw_path)
            safe_path.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if append_mode else "w"
            with safe_path.open(mode, encoding="utf-8") as handle:
                handle.write(content)
            action = "appended" if append_mode else "written"
            return f"file_write: successfully {action} {len(content)} chars to {safe_path}"
        except Exception as exc:
            return f"file_write error: {exc}"

    def _resolve_safe_path(self, raw_path: str) -> Path:
        path = Path(raw_path)
        resolved = (path if path.is_absolute() else (self.base_dir / path)).resolve()
        if self.base_dir not in resolved.parents and resolved != self.base_dir:
            raise ValueError("path is outside allowed base directory")
        return resolved
