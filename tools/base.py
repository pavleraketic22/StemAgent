from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    """Base contract for all runtime tools."""

    name: str
    description: str

    @abstractmethod
    def run(self, context: dict[str, Any]) -> str:
        """Execute tool from agent context and return text output."""
