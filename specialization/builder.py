from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from specialization.architect import ArchitecturePlan

SKILL_WRITER_PROMPT = """
You are writing a skill file for an AI agent pipeline step.

Task class: {task_class}
Pipeline step: {step}
Step goal: {brief}
Full pipeline (for context): {pipeline_steps}
Available tools: {tools}

Write a complete skill prompt that this agent will use at runtime.
The skill will receive these runtime variables: task_class, question, selected_tools.
Previous step outputs will be available as: {prior_step_vars}

The skill prompt must:
1. Tell the agent exactly HOW to do this step (not just what)
2. Be specific to {task_class} — not generic
3. Specify what good output looks like
4. Name failure modes to avoid

Return ONLY the skill prompt text. No explanation, no markdown wrapper.
"""


class Builder:
    def __init__(self, project_root: Path | None = None) -> None:
        load_dotenv()
        self.project_root = (project_root or Path.cwd()).resolve()
        self.skills_dir = self.project_root / "skills"
        self.config_path = self.project_root / "agents" / "agent_config.json"

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in .env")
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"

    def _write_skill_with_llm(
        self,
        step: str,
        spec: dict,
        plan: ArchitecturePlan,
        prior_steps: list[str],
    ) -> str:
        prior_vars = (
            "\n".join(f"{{{s}}}" for s in prior_steps)
            if prior_steps else "none (this is the first step)"
        )
        prompt = SKILL_WRITER_PROMPT.format(
            task_class=plan.task_class,
            step=step,
            brief=spec.get("brief", ""),
            pipeline_steps=" → ".join(plan.pipeline_steps),
            tools=", ".join(plan.tools) if plan.tools else "none",
            prior_step_vars=prior_vars,
        )
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": "You write precise skill prompts for AI agents."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    def build(self, plan, *, config_path=None, skills_dir=None) -> None:
        active_skills_dir = (skills_dir or self.skills_dir).resolve()
        active_skills_dir.mkdir(parents=True, exist_ok=True)

        skills_config = {}
        completed_steps = []

        for step in plan.pipeline_steps:
            spec = plan.skill_specs.get(step, {"brief": step})

            # LLM writes the skill — knows what came before
            skill_content = self._write_skill_with_llm(
                step=step,
                spec=spec,
                plan=plan,
                prior_steps=completed_steps,
            )

            prompt_file = active_skills_dir / f"{step}.md"
            prompt_file.write_text(skill_content, encoding="utf-8")

            skills_config[step] = {
                "name": spec.get("name", step),
                "tools": spec.get("tools", ["auto"]),
                "prompt_file": str(prompt_file.relative_to(self.project_root)),
            }
            completed_steps.append(step)

        config = {
            "task_class": plan.task_class,
            "architecture": "pipeline",
            "model": self.model,
            "system_prompt": plan.system_prompt,
            "tools": plan.tools,
            "pipeline_steps": plan.pipeline_steps,
            "skills": skills_config,
        }
        self.write_config(config, config_path=config_path)

    def read_current_config(self, config_path: Path | str | None = None) -> dict:
        if config_path is None:
            path = self.config_path
        elif isinstance(config_path, str):
            path = Path(config_path)
        else:
            path = config_path
        
        path = path.resolve()
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def write_config(self, config: dict, config_path: Path | str | None = None) -> None:
        if config_path is None:
            path = self.config_path
        elif isinstance(config_path, str):
            path = Path(config_path)
        else:
            path = config_path
        
        path = path.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")