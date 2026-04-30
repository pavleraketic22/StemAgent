from __future__ import annotations

from pathlib import Path
import shutil

from agents.agent import Agent
from eval.benchmark_runner import BenchmarkRunner
from specialization.pipeline import SpecializationPipeline


def _normalize_task_class(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"research", "deep research", "deep_research"}:
        return "Deep Research"
    if normalized in {"qa", "quality assurance", "quality"}:
        return "QA"
    if normalized in {"security", "sec"}:
        return "Security"
    return value.strip() or "Deep Research"


def _normalize_mode(value: str) -> str:
    normalized = value.strip().lower()
    allowed = {"execute", "specialize", "evolve", "benchmark"}
    return normalized if normalized in allowed else "execute"


def _ensure_stem_config() -> tuple[Path, Path]:
    stem = Path("agents/agent_config.stem.json")
    live = Path("agents/agent_config.json")
    if not stem.exists() and live.exists():
        stem.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(live, stem)
    return stem, live


def _reset_live_config_to_stem(stem: Path, live: Path) -> None:
    if stem.exists():
        live.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(stem, live)


def _chat_loop(task_class: str, config_path: str) -> bool:
    """Returns True if app should exit, False to go back."""
    agent = Agent(config_path)
    print("\nYou can now ask multiple questions. Commands: /back, /exit")
    while True:
        query = input("Query: ").strip()
        if not query:
            continue

        if query.lower() == "/back":
            return False
        if query.lower() == "/exit":
            return True

        result = agent.run(question=query, task_class=task_class)
        print(f"\nTask class: {result['task_class']}")
        print(
            f"Selected tools: {', '.join(result['selected_tools']) if result['selected_tools'] else 'none'}"
        )
        print(f"\nAnswer:\n{result['answer']}\n")


def main() -> None:
    stem_path, live_path = _ensure_stem_config()
    _reset_live_config_to_stem(stem_path, live_path)
    pipeline = SpecializationPipeline()

    active_session_dirs: list[Path] = []

    try:
        while True:
            raw_task = input("Enter task class (Deep Research / QA / Security) or /exit: ").strip()
            if raw_task.lower() == "/exit":
                break

            task_class = _normalize_task_class(raw_task)
            mode = _normalize_mode(
                input("Mode (execute/specialize/evolve/benchmark) [execute]: ") or "execute"
            )

            if mode == "benchmark":
                runner = BenchmarkRunner(config_path=str(stem_path))
                bench = runner.run()
                print("\nBenchmark complete.")
                print(f"Baseline avg total: {bench.baseline_average_total:.2f}/25")
                print(f"Specialized avg total: {bench.specialized_average_total:.2f}/25")
                print(f"Delta total: {bench.delta_total:+.2f}")
                print(f"Comparative win rate: {bench.comparative_win_rate:.3f}")
                print("Baseline by domain:")
                for domain, score in bench.baseline_by_domain.items():
                    print(f"- {domain}: {score:.2f}/25")
                print("Specialized by domain:")
                for domain, score in bench.specialized_by_domain.items():
                    print(f"- {domain}: {score:.2f}/25")
                print("Specialized by difficulty:")
                for difficulty, score in bench.specialized_by_difficulty.items():
                    print(f"- {difficulty}: {score:.2f}/25")
                continue

            question = input("Enter question: ").strip()
            if not question:
                print("Question is required for this mode.")
                continue

            if mode == "execute":
                should_exit = _chat_loop(task_class=task_class, config_path=str(stem_path))
                if should_exit:
                    break
                continue

            session_dir, session_skills_dir, session_config_path = pipeline.create_session_paths(task_class)
            active_session_dirs.append(session_dir)

            if mode == "specialize":
                specialization_result = pipeline.run(
                    task_class=task_class,
                    dry_question=question,
                    config_path=session_config_path,
                    skills_dir=session_skills_dir,
                )
                eval_payload = specialization_result["evaluation"]
                print("\nSpecialization complete.")
                print(f"Score: {eval_payload['score']:.2f}")
                print(f"Should stop: {eval_payload['should_stop']}")
                print("Reasons:")
                for reason in eval_payload["reasons"]:
                    print(f"- {reason}")

                print("\nInitial answer on your query:")
                initial_agent = Agent(str(session_config_path))
                initial_result = initial_agent.run(question=question, task_class=task_class)
                print(f"\nAnswer:\n{initial_result['answer']}\n")

                should_exit = _chat_loop(task_class=task_class, config_path=str(session_config_path))
                if should_exit:
                    break

                pipeline.cleanup_session(session_dir)
                active_session_dirs = [d for d in active_session_dirs if d != session_dir]
                continue

            if mode == "evolve":
                evolution_result = pipeline.evolve(
                    task_class=task_class,
                    dry_question=question,
                    max_iterations=3,
                    stop_threshold=0.78,
                    config_path=session_config_path,
                    skills_dir=session_skills_dir,
                )
                print("\nEvolution complete.")
                print(f"Best iteration: {evolution_result['best_iteration']}")
                print(f"Best score: {evolution_result['best_score']:.3f}")
                before_avg = evolution_result["before_benchmark"]["specialized_average_total"]
                after_avg = evolution_result["after_benchmark"]["specialized_average_total"]
                print(f"Before benchmark avg: {before_avg:.2f}/25")
                print(f"After benchmark avg: {after_avg:.2f}/25")
                print(f"Report: {evolution_result['report_path']}")

                print("\nInitial answer on your query:")
                initial_agent = Agent(str(session_config_path))
                initial_result = initial_agent.run(question=question, task_class=task_class)
                print(f"\nAnswer:\n{initial_result['answer']}\n")

                should_exit = _chat_loop(task_class=task_class, config_path=str(session_config_path))
                if should_exit:
                    break

                pipeline.cleanup_session(session_dir)
                active_session_dirs = [d for d in active_session_dirs if d != session_dir]
                continue
    finally:
        for session_dir in active_session_dirs:
            pipeline.cleanup_session(session_dir)
        _reset_live_config_to_stem(stem_path, live_path)


if __name__ == "__main__":
    main()
