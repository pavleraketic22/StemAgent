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


# Domain-specific pipeline templates (stable skeleton)
DOMAIN_PIPELINE_TEMPLATES: dict[str, list[str]] = {
    "Deep Research": ["explore", "analyze", "synthesize", "cite"],
    "QA": ["understand_requirements", "design_tests", "validate_coverage"],
    "Security": ["identify_threats", "assess_risks", "propose_mitigations"],
}

# Default tools per domain
DOMAIN_DEFAULT_TOOLS: dict[str, list[str]] = {
    "Deep Research": ["web_search", "arxiv_search", "wikipedia_search", "citation_extractor"],
    "QA": ["file_read", "checklist_validator", "diff_compare", "text_chunker"],
    "Security": ["web_search", "security_pattern_scan", "file_read", "diff_compare"],
}


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

    def run(self, task_class: str, question: str | None = None) -> ExplorerResult:
        """Run exploration with optional question for focus.
        
        Args:
            task_class: Domain (Deep Research, QA, Security)
            question: Optional question to focus skill briefs
        """
        available_tools = sorted(AVAILABLE_TOOLS.keys())
        
        # Use domain template as base (stable pipeline)
        template_steps = DOMAIN_PIPELINE_TEMPLATES.get(task_class, ["explore", "analyze", "answer"])
        default_tools = DOMAIN_DEFAULT_TOOLS.get(task_class, ["web_search", "file_read"])
        
        # Generate skill briefs - either from question focus or general
        if question:
            skill_briefs = self._generate_focused_briefs(template_steps, task_class, question)
        else:
            skill_briefs = self._generate_generic_briefs(template_steps, task_class)
        
        # Get approach summary from web evidence (for context)
        approach_summary = self._get_approach_summary(task_class)
        
        return ExplorerResult(
            task_class=task_class,
            approach_summary=approach_summary,
            recommended_pipeline_steps=template_steps,
            recommended_tools=default_tools,
            skill_briefs=skill_briefs,
        )

    def _generate_focused_briefs(self, steps: list[str], task_class: str, question: str) -> dict[str, str]:
        """Generate skill briefs focused on the specific question."""
        prompt = (
            "Generate brief skill descriptions for a specialized agent.\n"
            f"Task class: {task_class}\n"
            f"Question: {question}\n"
            f"Pipeline steps: {steps}\n\n"
            "For each step, write a 1-sentence brief that focuses on how to handle this specific question.\n"
            "Return ONLY valid JSON:\n"
            '{"step_name": "brief description", ...}'
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": "You generate focused skill briefs."},
                    {"role": "user", "content": prompt},
                ],
            )
            raw = (response.choices[0].message.content or "{}").strip()
            briefs = json.loads(self._extract_json_text(raw))
            return {str(k): str(v) for k, v in briefs.items()}
        except Exception:
            # Fallback to generic briefs
            return self._generate_generic_briefs(steps, task_class)

    def _generate_generic_briefs(self, steps: list[str], task_class: str) -> dict[str, str]:
        """Generate generic skill briefs for the domain."""
        briefs = {}
        for step in steps:
            if step == "explore":
                briefs[step] = f"Gather relevant information and sources for {task_class} tasks."
            elif step == "analyze":
                briefs[step] = f"Analyze gathered information with {task_class} expertise."
            elif step == "synthesize":
                briefs[step] = f"Synthesize findings into coherent {task_class} output."
            elif step == "cite":
                briefs[step] = "Include proper citations and references."
            elif step == "understand_requirements":
                briefs[step] = "Understand and clarify requirements thoroughly."
            elif step == "design_tests":
                briefs[step] = "Design comprehensive test cases and scenarios."
            elif step == "validate_coverage":
                briefs[step] = "Validate test coverage and identify gaps."
            elif step == "identify_threats":
                briefs[step] = "Identify potential security threats and vulnerabilities."
            elif step == "assess_risks":
                briefs[step] = "Assess risk severity and likelihood."
            elif step == "propose_mitigations":
                briefs[step] = "Propose layered security mitigations."
            else:
                briefs[step] = f"Perform {step} for {task_class} tasks."
        return briefs

    def _get_approach_summary(self, task_class: str) -> str:
        """Get approach summary from web evidence."""
        prompt = (
            f"You are a {task_class} expert. In 2-3 sentences, describe the core approach "
            "and mindset needed for high-quality work in this domain."
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": "You are concise."},
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content or f"Specialized {task_class} agent"
        except Exception:
            return f"Specialized {task_class} agent"

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
