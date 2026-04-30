from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

try:
    from tools.tool_library import AVAILABLE_TOOLS
except Exception:
    AVAILABLE_TOOLS: dict[str, Any] = {}


@dataclass
class ExplorerResult:
    task_class: str
    approach_summary: str
    recommended_pipeline_steps: list[str]
    recommended_tools: list[str]
    skill_briefs: dict[str, str]


class Explorer:
    """Explorer researches how a task class is typically solved."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in .env")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def run(self, task_class: str) -> ExplorerResult:
        available_tools = sorted(AVAILABLE_TOOLS.keys())
        discovery_context = self._gather_open_web_evidence(task_class)

        prompt = (
            "You are an expert workflow researcher for autonomous agents.\n"
            "Task class: {task_class}\n"
            "Available tools:\n{available_tools}\n\n"
            "Open-web evidence snippets:\n{evidence}\n\n"
            "Produce ONLY JSON with keys:\n"
            "approach_summary (string),\n"
            "recommended_pipeline_steps (list[string]),\n"
            "recommended_tools (list[string], subset of available tools),\n"
            "skill_briefs (object: step -> concise skill brief).\n"
            "No markdown, no commentary."
        ).format(
            task_class=task_class,
            available_tools="\n".join(f"- {name}" for name in available_tools),
            evidence=discovery_context,
        )

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": "Return strict JSON only."},
                {"role": "user", "content": prompt},
            ],
        )

        raw = (response.choices[0].message.content or "{}").strip()
        payload = json.loads(self._extract_json_text(raw))

        tools = payload.get("recommended_tools", [])
        clean_tools = [t for t in tools if isinstance(t, str) and t in available_tools]

        steps = payload.get("recommended_pipeline_steps", [])
        clean_steps = [s for s in steps if isinstance(s, str) and s.strip()]
        if not clean_steps:
            clean_steps = ["answer"]

        briefs = payload.get("skill_briefs", {})
        if not isinstance(briefs, dict):
            briefs = {}

        return ExplorerResult(
            task_class=task_class,
            approach_summary=str(payload.get("approach_summary", "")),
            recommended_pipeline_steps=clean_steps,
            recommended_tools=clean_tools,
            skill_briefs={str(k): str(v) for k, v in briefs.items()},
        )

    def _gather_open_web_evidence(self, task_class: str) -> str:
        snippets: list[str] = []

        web_search = AVAILABLE_TOOLS.get("web_search")
        if web_search:
            snippets.append(
                "[web_search]\n"
                + web_search.run(
                    {
                        "search_query": (
                            f"best workflow steps for AI agent {task_class} tasks "
                            "methodology checklist"
                        ),
                        "search_num": 5,
                    }
                )
            )

        wikipedia = AVAILABLE_TOOLS.get("wikipedia_search")
        if wikipedia:
            snippets.append(
                "[wikipedia_search]\n"
                + wikipedia.run({"wikipedia_query": task_class, "wikipedia_limit": 3})
            )

        arxiv = AVAILABLE_TOOLS.get("arxiv_search")
        if arxiv:
            snippets.append(
                "[arxiv_search]\n"
                + arxiv.run({"arxiv_query": task_class, "arxiv_max_results": 3})
            )

        if not snippets:
            return "No external evidence tools available."
        return "\n\n".join(snippets)

    @staticmethod
    def _extract_json_text(text: str) -> str:
        value = text.strip()
        if value.startswith("```") and value.endswith("```"):
            lines = value.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            value = "\n".join(lines).strip()
            if value.lower().startswith("json"):
                value = value[4:].strip()
        return value
