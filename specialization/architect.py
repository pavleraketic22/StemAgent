from __future__ import annotations

import json
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

from specialization.explorer import ExplorerResult


@dataclass
class ArchitecturePlan:
    task_class: str
    system_prompt: str
    pipeline_steps: list[str]
    tools: list[str]
    skill_specs: dict[str, dict[str, object]]


ARCHITECT_PROMPT = """
You are designing a specialized AI agent for the task class: {task_class}

Explorer findings:
- Approach summary: {approach_summary}
- Recommended pipeline steps: {pipeline_steps}
- Recommended tools: {tools}
- Skill briefs: {skill_briefs}

Your job is to produce a complete architecture plan. Think like a senior
engineer who has done {task_class} work for years — not like someone
writing a generic template.

Produce ONLY valid JSON with this exact shape:

{{
  "system_prompt": "...",
  "pipeline_steps": ["step1", "step2", ...],
  "tools": ["tool1", ...],
  "skill_specs": {{
    "step_name": {{
      "name": "Human readable name",
      "brief": "One sentence goal",
      "instructions": [
        "Concrete instruction 1",
        "Concrete instruction 2"
      ],
      "output_format": "Exact description of what this step should return",
      "failure_modes": ["Specific thing to avoid 1", "..."],
      "tools": ["auto"]
    }}
  }}
}}

Rules:
- system_prompt must be specific to {task_class} — name the domain,
  name the mindset, name what good output looks like in this domain
- pipeline_steps must reflect actual expert workflow for {task_class},
  not generic steps like "answer" or "respond"
- each skill's instructions must be actionable and domain-specific —
  a {task_class} practitioner reading this should recognize the workflow
- failure_modes must name real mistakes that happen in {task_class} work,
  not generic AI failure modes
- output_format must describe the exact structure a downstream step
  or human would expect

No markdown. No explanation. Only JSON.
"""


class Architect:
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in .env")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def build_plan(self, exploration: ExplorerResult) -> ArchitecturePlan:
        prompt = ARCHITECT_PROMPT.format(
            task_class=exploration.task_class,
            approach_summary=exploration.approach_summary or "Not provided",
            pipeline_steps=exploration.recommended_pipeline_steps,
            tools=exploration.recommended_tools,
            skill_briefs=json.dumps(exploration.skill_briefs, indent=2),
        )

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You design precise AI agent architectures. "
                        "Return strict JSON only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )

        raw = (response.choices[0].message.content or "{}").strip()
        raw = self._extract_json(raw)

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            # Fallback na stari deterministički pristup
            return self._fallback_plan(exploration)

        return self._validate_and_build(payload, exploration)

    def _validate_and_build(
        self, payload: dict, exploration: ExplorerResult
    ) -> ArchitecturePlan:
        # Pipeline steps — nikad prazan
        steps = payload.get("pipeline_steps", [])
        if not steps or not all(isinstance(s, str) for s in steps):
            steps = exploration.recommended_pipeline_steps or ["answer"]

        # Tools — samo oni koji postoje u exploration
        raw_tools = payload.get("tools", [])
        valid_tools = [
            t for t in raw_tools
            if isinstance(t, str)
        ] or exploration.recommended_tools

        # skill_specs — svaki step mora imati entry
        raw_specs = payload.get("skill_specs", {})
        skill_specs: dict[str, dict] = {}
        for step in steps:
            spec = raw_specs.get(step, {})
            skill_specs[step] = {
                "name": spec.get("name", step.replace("_", " ").title()),
                "brief": spec.get(
                    "brief",
                    exploration.skill_briefs.get(step, f"Perform {step}"),
                ),
                "instructions": spec.get("instructions", []),
                "output_format": spec.get("output_format", "Structured prose"),
                "failure_modes": spec.get("failure_modes", []),
                "tools": spec.get("tools", ["auto"]),
                "prompt_file": f"skills/{step}.md",
            }

        # system_prompt — ne sme biti generičan
        system_prompt = str(payload.get("system_prompt", ""))
        if not system_prompt or len(system_prompt) < 50:
            system_prompt = self._default_system_prompt(exploration)

        return ArchitecturePlan(
            task_class=exploration.task_class,
            system_prompt=system_prompt,
            pipeline_steps=steps,
            tools=valid_tools,
            skill_specs=skill_specs,
        )

    def _fallback_plan(self, exploration: ExplorerResult) -> ArchitecturePlan:
        """Deterministički fallback ako LLM vrati invalid JSON."""
        steps = exploration.recommended_pipeline_steps or ["answer"]
        skill_specs = {
            step: {
                "name": step.replace("_", " ").title(),
                "brief": exploration.skill_briefs.get(step, f"Perform {step}"),
                "instructions": [],
                "output_format": "Structured prose",
                "failure_modes": [],
                "tools": ["auto"],
                "prompt_file": f"skills/{step}.md",
            }
            for step in steps
        }
        return ArchitecturePlan(
            task_class=exploration.task_class,
            system_prompt=self._default_system_prompt(exploration),
            pipeline_steps=steps,
            tools=exploration.recommended_tools,
            skill_specs=skill_specs,
        )

    @staticmethod
    def _default_system_prompt(exploration: ExplorerResult) -> str:
        return (
            f"You are a specialized {exploration.task_class} agent.\n"
            f"{exploration.approach_summary or ''}\n"
            "Use tool evidence. State uncertainty explicitly. Be concise."
        )

    @staticmethod
    def _extract_json(text: str) -> str:
        value = text.strip()
        if value.startswith("```"):
            lines = value.splitlines()
            lines = lines[1:] if lines[0].startswith("```") else lines
            lines = lines[:-1] if lines and lines[-1].startswith("```") else lines
            value = "\n".join(lines).strip()
            if value.lower().startswith("json"):
                value = value[4:].strip()
        return value