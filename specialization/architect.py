from __future__ import annotations

from dataclasses import dataclass

from specialization.explorer import ExplorerResult


@dataclass
class ArchitecturePlan:
    """Concrete specialization plan for agent config + skills."""

    task_class: str
    system_prompt: str
    pipeline_steps: list[str]
    tools: list[str]
    skill_specs: dict[str, dict[str, object]]


class Architect:
    """Architect converts exploration findings into executable plan."""

    def build_plan(self, exploration: ExplorerResult) -> ArchitecturePlan:
        steps = exploration.recommended_pipeline_steps or ["answer"]

        skill_specs: dict[str, dict[str, object]] = {}
        for step in steps:
            brief = exploration.skill_briefs.get(
                step,
                f"Perform step '{step}' for task class {exploration.task_class}.",
            )
            skill_specs[step] = {
                "name": step.replace("_", " ").title(),
                "tools": ["auto"],
                "prompt_file": f"skills/{step}.md",
                "brief": brief,
            }

        system_prompt = (
            "You are a specialized {task_class} agent.\n"
            "Use evidence from tools, state uncertainty explicitly, and be concise.\n"
            "Workflow rationale: {summary}"
        ).format(
            task_class=exploration.task_class,
            summary=exploration.approach_summary or "Use best-practice staged reasoning.",
        )

        return ArchitecturePlan(
            task_class=exploration.task_class,
            system_prompt=system_prompt,
            pipeline_steps=steps,
            tools=exploration.recommended_tools,
            skill_specs=skill_specs,
        )
