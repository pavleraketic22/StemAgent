import json
import importlib
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

AVAILABLE_TOOLS: dict[str, Any] = {}
try:
    tool_library = importlib.import_module("tools.tool_library")
    AVAILABLE_TOOLS = getattr(tool_library, "AVAILABLE_TOOLS", {})
except Exception:
    # Tools are optional while the stem agent scaffold is evolving.
    pass


class Agent:
    def __init__(self, config_path: str):
        load_dotenv()

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Add it to your .env file."
            )

        self.client = OpenAI(api_key=api_key)

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.model = self.config.get("model", "gpt-4o-mini")
        self.system_prompt = self.config.get(
            "system_prompt", "You are a helpful research agent."
        )

        raw_skills = self.config.get("skills", {})
        if isinstance(raw_skills, list):
            # Backward compatibility: when skills is a list of names,
            # convert to empty templates so the app doesn't crash.
            self.skills = {name: {"prompt_template": "{question}"} for name in raw_skills}
        else:
            self.skills = raw_skills

        self.pipeline_steps = self.config.get("pipeline_steps", ["answer"])
        configured_tools = self.config.get("tools", [])
        self.configured_tool_allowlist: list[str] = (
            configured_tools if isinstance(configured_tools, list) else []
        )

        # Active toolset is selected at runtime by specialization step.
        self.tools: dict[str, Any] = {}

    def run(self, question: str, task_class: str | None = None) -> dict:
        selected_tool_names = self._select_tools_for_task(
            task_class=task_class or str(self.config.get("task_class", "")),
            question=question,
        )

        self.tools = {
            name: tool
            for name, tool in AVAILABLE_TOOLS.items()
            if name in selected_tool_names
        }

        context = {
            "question": question,
            "task_class": task_class or self.config.get("task_class", "Unknown"),
            "selected_tools": ", ".join(selected_tool_names),
        }

        for step in self.pipeline_steps:
            skill = self.skills.get(step, {"prompt_template": "{question}"})
            context[step] = self._execute_skill(skill, context)

        return {
            "question": question,
            "task_class": context["task_class"],
            "selected_tools": selected_tool_names,
            "answer": context[self.pipeline_steps[-1]],
            "steps": {s: context[s] for s in self.pipeline_steps}
        }

    def _select_tools_for_task(self, task_class: str, question: str) -> list[str]:
        available = {
            name: getattr(tool, "description", "")
            for name, tool in AVAILABLE_TOOLS.items()
        }

        if not available:
            return []

        allowlist = self.configured_tool_allowlist
        if allowlist:
            available = {k: v for k, v in available.items() if k in allowlist}

        if not available:
            return []

        planning_prompt = (
            "Select the best tools for this task class.\n"
            "Task class: {task_class}\n"
            "Question: {question}\n"
            "Available tools (name: description):\n{available_tools}\n\n"
            "Return ONLY valid JSON with this shape:\n"
            "{{\"tools\": [\"tool_name\", ...]}}\n"
            "Rules: choose 2-5 tools, only from available names, no explanation."
        ).format(
            task_class=task_class,
            question=question,
            available_tools="\n".join(
                f"- {name}: {desc}" for name, desc in sorted(available.items())
            ),
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a strict JSON planner."},
                    {"role": "user", "content": planning_prompt},
                ],
                temperature=0,
            )
            raw = response.choices[0].message.content or "{}"
            raw = self._extract_json_text(raw)
            parsed = json.loads(raw)
            selected = parsed.get("tools", [])
            if isinstance(selected, list):
                clean = [t for t in selected if isinstance(t, str) and t in available]
                if clean:
                    return clean[:5]
        except Exception:
            pass

        return self._fallback_select_tools(task_class, question, list(available.keys()))

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

    @staticmethod
    def _fallback_select_tools(
        task_class: str, question: str, available_names: list[str]
    ) -> list[str]:
        text = f"{task_class} {question}".lower()

        candidates: list[str] = []
        if any(k in text for k in ["deep research", "research", "paper", "literature"]):
            candidates.extend(
                [
                    "web_search",
                    "url_fetch",
                    "arxiv_search",
                    "wikipedia_search",
                    "citation_extractor",
                ]
            )
        if any(k in text for k in ["qa", "quality", "review", "check"]):
            candidates.extend(
                [
                    "file_read",
                    "diff_compare",
                    "checklist_validator",
                    "text_chunker",
                ]
            )
        if any(k in text for k in ["security", "vulnerability", "secure", "owasp"]):
            candidates.extend(
                [
                    "file_read",
                    "security_pattern_scan",
                    "diff_compare",
                    "web_search",
                ]
            )

        if not candidates:
            candidates = ["web_search", "file_read", "text_chunker"]

        dedup: list[str] = []
        for name in candidates:
            if name in available_names and name not in dedup:
                dedup.append(name)
        return dedup[:5]

    def _execute_skill(self, skill: dict, context: dict) -> str:
        # popuni prompt_template sa kontekstom
        prompt_template = self._load_skill_prompt(skill)
        prompt = self._safe_format(prompt_template, context)

        # skill can request single tool, multiple tools, or 'auto' for selected toolset
        requested: list[str] = []
        single_tool = skill.get("tool")
        if isinstance(single_tool, str):
            requested.append(single_tool)
        multi_tools = skill.get("tools")
        if isinstance(multi_tools, list):
            requested.extend([t for t in multi_tools if isinstance(t, str)])

        resolved_tools: list[str] = []
        for tool_name in requested:
            if tool_name == "auto":
                resolved_tools.extend(self.tools.keys())
            elif tool_name in self.tools:
                resolved_tools.append(tool_name)

        # de-duplicate while preserving order
        unique_tools: list[str] = []
        for name in resolved_tools:
            if name not in unique_tools:
                unique_tools.append(name)

        if unique_tools:
            tool_outputs: list[str] = []
            for tool_name in unique_tools:
                try:
                    output = self.tools[tool_name].run(context)
                    tool_outputs.append(f"[{tool_name}]\n{output}")
                except Exception as exc:
                    tool_outputs.append(f"[{tool_name}] error: {exc}")
            prompt += "\n\nTool results:\n" + "\n\n".join(tool_outputs)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content or ""

    @staticmethod
    def _safe_format(template: str, context: dict[str, Any]) -> str:
        class _SafeDict(dict[str, Any]):
            def __missing__(self, key: str) -> str:
                return "{" + key + "}"

        return template.format_map(_SafeDict(context))

    @staticmethod
    def _load_skill_prompt(skill: dict[str, Any]) -> str:
        prompt_file = skill.get("prompt_file")
        if isinstance(prompt_file, str) and prompt_file.strip():
            path = Path(prompt_file)
            if path.exists() and path.is_file():
                return path.read_text(encoding="utf-8")
        return str(skill.get("prompt_template", "{question}"))
