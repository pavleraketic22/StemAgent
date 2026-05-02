from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import shutil

from agents.agent import Agent
from agents.skills.skill_manager import SkillManager
from eval.benchmark_data import BENCHMARK_CASES
from eval.benchmark_runner import BenchmarkRunner, to_dict
from specialization.architect import Architect
from specialization.builder import Builder
from specialization.evaluator import Evaluator
from specialization.explorer import Explorer


class SpecializationPipeline:
    """Simple v1 specialization loop: explorer -> architect -> builder -> evaluate."""

    def __init__(self) -> None:
        self.explorer = Explorer()
        self.architect = Architect()
        self.builder = Builder()
        self.evaluator = Evaluator()
        self.benchmark = BenchmarkRunner()
        self.results_dir = Path("eval/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def create_session_paths(task_class: str) -> tuple[Path, Path, Path]:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        safe_class = task_class.lower().replace(" ", "_")
        session_dir = Path("runs") / f"{safe_class}_{timestamp}"
        skills_dir = session_dir / "skills"
        config_path = session_dir / "agent_config.specialized.json"
        session_dir.mkdir(parents=True, exist_ok=True)
        skills_dir.mkdir(parents=True, exist_ok=True)
        return session_dir, skills_dir, config_path

    @staticmethod
    def cleanup_session(session_dir: Path) -> None:
        if session_dir.exists() and session_dir.is_dir():
            shutil.rmtree(session_dir, ignore_errors=True)
        if Path("skills/learnings.md").exists():
            shutil.rmtree("skills/learnings.md", ignore_errors=True)

    def _auto_learn(
        self,
        task_class: str,
        eval_result,
        question: str,
        answer: str,
        threshold: float = 0.75,
    ) -> None:
        """Extract and append learnings after each iteration to improve score."""
        score = eval_result.score

        # Always learn (no threshold) to ensure improvement
        # Use SkillManager to extract learning with LLM
        skill_manager = SkillManager()
        
        # Get dimension scores to find what to improve
        dim_scores = eval_result.dimension_scores if hasattr(eval_result, 'dimension_scores') else {}
        
        # Find weakest dimension
        weakest_dim = None
        weakest_score = 1.0
        if dim_scores:
            for dim, score_val in dim_scores.items():
                if score_val < weakest_score:
                    weakest_score = score_val
                    weakest_dim = dim
        
        # Generate improvement based on weakest dimension - use ACTUAL answer
        learning = skill_manager.extract_learning_with_llm(
            question=question,
            answer=answer,  # Use actual answer, not just reasoning
            scores=dim_scores,
            domain=task_class,
        )

        if learning:
            skill_manager.append_learning(
                domain=task_class,
                improvement=learning,
                question=question,
                scores=dim_scores,
            )
            print(f"\n[Auto-learn] {weakest_dim}: {learning[:80]}...")

    def run(
        self,
        task_class: str,
        dry_question: str = "Explain your reasoning process.",
        *,
        config_path: Path | None = None,
        skills_dir: Path | None = None,
        max_iterations: int = 5,
    ) -> dict:
        active_config = str(config_path) if config_path else "agents/agent_config.json"
        
        best_result = None
        best_score = 0.0
        history: list[dict] = []

        for iteration in range(1, max_iterations + 1):
            print(f"\n--- Iteration {iteration}/{max_iterations} ---")
            
            # Explore → Architect → Build
            exploration = self.explorer.run(task_class, question=dry_question)
            plan = self.architect.build_plan(exploration)
            self.builder.build(plan, config_path=active_config, skills_dir=skills_dir)

            # Run agent
            agent = Agent(active_config)
            result = agent.run(question=dry_question, task_class=task_class)

            # Evaluate
            eval_result = self.evaluator.evaluate(
                selected_tools=result.get("selected_tools", []),
                pipeline_steps=plan.pipeline_steps,
                sample_outputs=[result.get("answer", "")],
                target_keywords=[task_class],
                task_class=task_class,
                question=dry_question,
            )

            print(f"Score: {eval_result.score:.3f} | Should stop: {eval_result.should_stop}")
            
            # Auto-learning: extract and append learnings after each iteration
            self._auto_learn(
                task_class=task_class,
                eval_result=eval_result,
                question=dry_question,
                answer=result.get("answer", "")[:1000],
            )
            
            entry = {
                "iteration": iteration,
                "exploration": asdict(exploration),
                "architecture": asdict(plan),
                "evaluation": asdict(eval_result),
                "smoke_result": result,
            }
            history.append(entry)

            # Keep best result
            if eval_result.score > best_score:
                best_score = eval_result.score
                best_result = entry

            # Stop if evaluator says to stop
            if eval_result.should_stop:
                print(f"✓ Stopping at iteration {iteration} (score >= 0.75)")
                break

        # Restore best config (best agent)
        if best_result and config_path:
            best_config = best_result["architecture"]
            self.builder.write_config(best_config, config_path=config_path)
            print(f"\n✓ Restored best agent (score: {best_score:.3f})")

        return {
            "task_class": task_class,
            "total_iterations": len(history),
            "best_score": best_score,
            "history": history,
            "best_result": best_result,
        }
