from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class EvaluationResult:
    should_stop: bool
    score: float
    reasons: list[str]


class Evaluator:
    """Simple stop-condition evaluator for specialization loops.

    This is intentionally lightweight for v1. It checks structural and behavioral
    signals that indicate the agent is sufficiently specialized.
    """

    def evaluate(
        self,
        *,
        selected_tools: list[str],
        pipeline_steps: list[str],
        sample_outputs: list[str],
        target_keywords: list[str] | None = None,
    ) -> EvaluationResult:
        reasons: list[str] = []
        score = 0.0

        # 1) Structural specialization
        if len(selected_tools) >= 3:
            score += 0.35
            reasons.append("Has >=3 selected tools.")
            for i in range(len(selected_tools)):
                reasons.append(f"Tool {i}:{selected_tools[i]}")
        else:
            reasons.append("Too few tools selected (<3).")

        if len(pipeline_steps) >= 2:
            score += 0.25
            reasons.append("Pipeline has >=2 steps.")
        else:
            reasons.append("Pipeline too shallow (<2 steps).")

        # 2) Output quality proxy
        non_empty = [o for o in sample_outputs if isinstance(o, str) and o.strip()]
        if len(non_empty) == len(sample_outputs) and sample_outputs:
            score += 0.20
            reasons.append("All sample outputs are non-empty.")
        else:
            reasons.append("Some sample outputs are empty.")

        # 3) Domain alignment proxy
        if target_keywords:
            joined = "\n".join(non_empty).lower()
            matched = sum(1 for kw in target_keywords if kw.lower() in joined)
            coverage = matched / max(len(target_keywords), 1)
            score += 0.20 * coverage
            reasons.append(f"Keyword coverage: {coverage:.2f}")
        else:
            score += 0.10
            reasons.append("No target keywords provided; neutral domain score.")

        should_stop = score >= 0.75
        if should_stop:
            reasons.append("Stop condition reached (score >= 0.75).")
        else:
            reasons.append("Continue evolution (score < 0.75).")

        return EvaluationResult(should_stop=should_stop, score=score, reasons=reasons)
