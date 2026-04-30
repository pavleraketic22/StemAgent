from __future__ import annotations

from typing import Any

from .base import BaseTool


class ChecklistValidatorTool(BaseTool):
    """Simple QA validator: checks whether required points appear in text."""

    name = "checklist_validator"
    description = "Validate output text against required checklist items."

    def run(self, context: dict[str, Any]) -> str:
        text = str(context.get("text") or context.get("answer") or "")
        required = context.get("required_points")

        if not text:
            return "checklist_validator: missing 'text' or 'answer'."
        if not isinstance(required, list) or not required:
            return "checklist_validator: provide non-empty 'required_points' list."

        text_l = text.lower()
        present: list[str] = []
        missing: list[str] = []

        for item in required:
            point = str(item).strip()
            if not point:
                continue
            if point.lower() in text_l:
                present.append(point)
            else:
                missing.append(point)

        score = 0.0
        total = len(present) + len(missing)
        if total > 0:
            score = len(present) / total

        return (
            f"coverage_score={score:.2f}\n"
            f"present={present}\n"
            f"missing={missing}"
        )
