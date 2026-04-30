from __future__ import annotations

import difflib
from typing import Any

from .base import BaseTool


class DiffCompareTool(BaseTool):
    """Compare two text inputs and return unified diff."""

    name = "diff_compare"
    description = "Generate unified diff between two texts."

    def run(self, context: dict[str, Any]) -> str:
        left = str(context.get("left_text") or "")
        right = str(context.get("right_text") or "")
        if not left and not right:
            return "diff_compare: provide 'left_text' and/or 'right_text'."

        left_name = str(context.get("left_name") or "left")
        right_name = str(context.get("right_name") or "right")
        max_lines = int(context.get("diff_max_lines", 400))

        diff = difflib.unified_diff(
            left.splitlines(),
            right.splitlines(),
            fromfile=left_name,
            tofile=right_name,
            lineterm="",
        )
        diff_lines = list(diff)
        if not diff_lines:
            return "diff_compare: no differences."

        return "\n".join(diff_lines[:max_lines])
