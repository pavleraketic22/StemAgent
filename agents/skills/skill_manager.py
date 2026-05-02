from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


class SkillManager:
    """Manages agent skills: loading, filtering, and auto-learning."""

    def __init__(
        self,
        learnings_path: str = "skills/learnings.md",
        workflow_dir: str = "skills",
    ):
        self.learnings_path = Path(learnings_path)
        self.workflow_dir = Path(workflow_dir)
        self._learnings_cache: str | None = None

        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = None

    def load_learnings(self) -> str:
        """Load all learnings from learnings.md"""
        if self._learnings_cache is not None:
            return self._learnings_cache

        if not self.learnings_path.exists():
            self._learnings_cache = ""
            return ""

        content = self.learnings_path.read_text(encoding="utf-8")
        self._learnings_cache = content
        return content

    def get_relevant_learnings(self, domain: str) -> str:
        """Extract learnings relevant to a specific domain."""
        content = self._learnings_cache or self.load_learnings()

        if not content:
            return ""

        # Extract sections relevant to domain
        sections = content.split("## ")
        relevant = []

        domain_lower = domain.lower()
        for section in sections:
            if domain_lower in section.lower():
                relevant.append(section.strip())

        return "\n\n".join(relevant) if relevant else ""

    def load_workflow_skills(self, steps: list[str]) -> dict[str, str]:
        """Load workflow skill files for given pipeline steps."""
        skills = {}
        for step in steps:
            skill_file = self.workflow_dir / f"{step}.md"
            if skill_file.exists():
                skills[step] = skill_file.read_text(encoding="utf-8")
            else:
                skills[step] = ""
        return skills

    def append_learning(
        self,
        domain: str,
        improvement: str,
        question: str | None = None,
        answer: str | None = None,
        scores: dict[str, Any] | None = None,
    ) -> None:
        """Append a new learning to learnings.md"""
        timestamp = datetime.utcnow().isoformat()

        # Ensure parent directory exists
        self.learnings_path.parent.mkdir(parents=True, exist_ok=True)

        # Build learning block
        learning_block = f"""## Learned Improvement ({domain}) - {timestamp}

**Trigger:** {"Low scores: " + str(scores) if scores else "Manual trigger"}
{("**Question:** " + question[:200] + "...") if question else ""}

**Improvement:**
{improvement}

---
"""

        # Append to file
        with open(self.learnings_path, "a", encoding="utf-8") as f:
            f.write("\n" + learning_block)

        # Invalidate cache
        self._learnings_cache = None

    def extract_learning_with_llm(
        self,
        question: str,
        answer: str,
        scores: dict[str, Any],
        domain: str,
    ) -> str | None:
        """Use LLM to extract actionable learning from failed case."""
        if not self.client:
            return None

        prompt = f"""You are analyzing a failed agent execution to extract a learning.

Domain: {domain}
Question: {question}

Answer given: {answer[:500]}...

Scores: {json.dumps(scores, indent=2)}

What concrete improvement rule should the agent follow next time?
Focus on 1-2 specific, actionable rules that would improve the score.

Return ONLY the rule as a bullet point starting with "- ".
Example: "- Always include edge cases explicitly in QA test plans."
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                messages=[
                    {"role": "system", "content": "You extract actionable learning rules."},
                    {"role": "user", "content": prompt},
                ],
            )
            result = response.choices[0].message.content or ""
            return result.strip() if result.strip() else None
        except Exception:
            return None

    def get_all_context(self, domain: str, pipeline_steps: list[str]) -> dict[str, str]:
        """Get all relevant context for agent: learnings + workflow skills."""
        return {
            "learnings": self.get_relevant_learnings(domain),
            "workflow": self.load_workflow_skills(pipeline_steps),
        }

    def clear_cache(self) -> None:
        """Clear learnings cache to force reload."""
        self._learnings_cache = None