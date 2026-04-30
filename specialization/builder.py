from __future__ import annotations

import json
from pathlib import Path

from specialization.architect import ArchitecturePlan


class Builder:
    """Builder materializes plan into skills markdown + agent config."""

    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = (project_root or Path.cwd()).resolve()
        self.skills_dir = self.project_root / "skills"
        self.config_path = self.project_root / "agents" / "agent_config.json"

    def read_current_config(self, config_path: Path | None = None) -> dict[str, object]:
        path = (config_path or self.config_path).resolve()
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def write_config(self, config: dict[str, object], config_path: Path | None = None) -> None:
        path = (config_path or self.config_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    def build(
        self,
        plan: ArchitecturePlan,
        *,
        config_path: Path | None = None,
        skills_dir: Path | None = None,
    ) -> None:
        active_skills_dir = (skills_dir or self.skills_dir).resolve()
        active_skills_dir.mkdir(parents=True, exist_ok=True)

        skills_config: dict[str, dict[str, object]] = {}
        for step, spec in plan.skill_specs.items():
            default_prompt_path = active_skills_dir / f"{step}.md"
            prompt_file = default_prompt_path
            if spec.get("prompt_file"):
                candidate = Path(str(spec["prompt_file"]))
                prompt_file = (
                    candidate.resolve()
                    if candidate.is_absolute()
                    else (self.project_root / candidate).resolve()
                )

            prompt_file.parent.mkdir(parents=True, exist_ok=True)
            prompt_file.write_text(
                self._render_skill_prompt(step=step, brief=str(spec.get("brief", ""))),
                encoding="utf-8",
            )

            skills_config[step] = {
                "name": spec.get("name", step),
                "tools": spec.get("tools", ["auto"]),
                "prompt_file": str(prompt_file.relative_to(self.project_root)),
            }

        config = {
            "task_class": plan.task_class,
            "architecture": "pipeline",
            "model": "gpt-4o-mini",
            "system_prompt": plan.system_prompt,
            "tools": plan.tools,
            "pipeline_steps": plan.pipeline_steps,
            "skills": skills_config,
        }
        self.write_config(config, config_path=config_path)

    @staticmethod
    def _render_skill_prompt(step: str, brief: str) -> str:
        return (
            f"# Skill: {step}\n\n"
            f"Goal: {brief}\n\n"
            "Task class: {task_class}\n"
            "Question: {question}\n"
            "Selected tools: {selected_tools}\n\n"
            "Use relevant tool evidence if present.\n"
            "If evidence is missing or weak, say so explicitly.\n"
            "Return concise, structured output for this step.\n"
        )
