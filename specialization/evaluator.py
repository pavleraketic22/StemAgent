from __future__ import annotations

import json
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI


@dataclass
class EvaluationResult:
    should_stop: bool
    score: float
    reasons: list[str]
    dimension_scores: dict[str, float]


JUDGE_PROMPT = """
You are evaluating whether an AI agent is sufficiently specialized
for the task class: {task_class}

Agent configuration:
- Pipeline steps: {pipeline_steps}
- Selected tools: {selected_tools}

Sample output from agent:
---
{sample_output}
---

IMPORTANT: First check if the answer actually ANSWERS the question.
If the answer doesn't address what was asked, score everything LOW.

Rate each dimension 0.0 to 1.0:

1. TASK_SPECIFICITY
   Does the output show domain expertise AND directly answer what was asked?
   0.0 = doesn't answer the question or generic response
   1.0 = clearly answers the question with domain expertise

2. REASONING_QUALITY
   Does the agent show its reasoning, not just conclusions?
   0.0 = bare assertions with no justification
   1.0 = clear reasoning chain with evidence and uncertainty marked

3. TOOL_UTILIZATION
   Are the selected tools appropriate for {task_class}?
   0.0 = wrong tools or no tools
   1.0 = exactly the right tools for this domain

4. OUTPUT_STRUCTURE
   Is the output structured for {task_class} consumption?
   0.0 = unstructured prose
   1.0 = format that an expert in {task_class} would expect

5. FAILURE_AVOIDANCE
   Does the output avoid common {task_class} failure modes
   (e.g. hallucination, shallow coverage, missing edge cases)?
   0.0 = multiple obvious failure modes present
   1.0 = no obvious failure modes

Return ONLY valid JSON:
{{
  "task_specificity": 0.0,
  "reasoning_quality": 0.0,
  "tool_utilization": 0.0,
  "output_structure": 0.0,
  "failure_avoidance": 0.0,
  "reasoning": "One sentence explaining the biggest strength and weakness"
}}
"""

# Weights per dimension — sum = 1.0
DIMENSION_WEIGHTS = {
    "task_specificity": 0.30,
    "reasoning_quality": 0.25,
    "tool_utilization": 0.20,
    "output_structure": 0.15,
    "failure_avoidance": 0.10,
}

# Structural checks — fast, no LLM needed
STRUCTURAL_FLOOR = {
    "min_tools": 1,
    "min_steps": 1,
    "min_output_len": 50,
}


class Evaluator:
    def __init__(self, model: str = "gpt-5.4-mini") -> None:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in .env")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def evaluate(
        self,
        *,
        selected_tools: list[str],
        pipeline_steps: list[str],
        sample_outputs: list[str],
        target_keywords: list[str] | None = None,
        task_class: str = "Unknown",
        question: str = "",
    ) -> EvaluationResult:
        reasons: list[str] = []

        # ── 1. Structural floor check ─────────────────────────────────────────
        structural_fail = self._check_structural_floor(
            selected_tools, pipeline_steps, sample_outputs, reasons
        )
        if structural_fail:
            return EvaluationResult(
                should_stop=False,
                score=0.0,
                reasons=reasons,
                dimension_scores={},
            )

        # ── 1.5. Quick relevance check ────────────────────────────────────────
        best_output = max(sample_outputs, key=len) if sample_outputs else ""
        relevance_penalty = self._check_answer_relevance(question, best_output)
        if relevance_penalty > 0:
            reasons.append(f"Relevance penalty: {relevance_penalty:.2f} (answer doesn't match question)")

        # ── 2. LLM judge ──────────────────────────────────────────────────────
        dim_scores = self._llm_judge(
            task_class=task_class,
            pipeline_steps=pipeline_steps,
            selected_tools=selected_tools,
            sample_output=best_output,
            reasons=reasons,
        )

        # ── 3. Weighted score ─────────────────────────────────────────────────
        score = sum(
            dim_scores.get(dim, 0.0) * weight
            for dim, weight in DIMENSION_WEIGHTS.items()
        )
        
        # Apply relevance penalty
        score = max(0, score - relevance_penalty)

        # ── 4. Bonus: keyword coverage (lightweight signal) ───────────────────
        if target_keywords:
            joined = best_output.lower()
            coverage = sum(
                1 for kw in target_keywords if kw.lower() in joined
            ) / max(len(target_keywords), 1)
            score = min(1.0, score + 0.05 * coverage)
            reasons.append(f"Keyword coverage bonus: {coverage:.2f}")

        should_stop = score >= 0.5
        reasons.append(
            f"Final score: {score:.3f} → "
            f"{'stop' if should_stop else 'continue'}"
        )

        return EvaluationResult(
            should_stop=should_stop,
            score=score,
            reasons=reasons,
            dimension_scores=dim_scores,
        )

    def _check_structural_floor(
        self,
        tools: list[str],
        steps: list[str],
        outputs: list[str],
        reasons: list[str],
    ) -> bool:
        """Returns True if agent fails structural minimums."""
        failed = False

        if len(tools) < STRUCTURAL_FLOOR["min_tools"]:
            reasons.append(f"FAIL: no tools selected")
            failed = True

        if len(steps) < STRUCTURAL_FLOOR["min_steps"]:
            reasons.append(f"FAIL: no pipeline steps")
            failed = True

        non_empty = [
            o
            for o in outputs
            if isinstance(o, str) and len(o.strip()) >= STRUCTURAL_FLOOR["min_output_len"]
        ]
        if not non_empty:
            reasons.append(
                f"FAIL: output empty or too short (<{STRUCTURAL_FLOOR['min_output_len']} chars)"
            )
            failed = True

        return failed

    def _llm_judge(
        self,
        task_class: str,
        pipeline_steps: list[str],
        selected_tools: list[str],
        sample_output: str,
        reasons: list[str],
    ) -> dict[str, float]:
        prompt = JUDGE_PROMPT.format(
            task_class=task_class,
            pipeline_steps=", ".join(pipeline_steps),
            selected_tools=", ".join(selected_tools),
            sample_output=sample_output[:2000],
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a strict evaluator. Return JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            raw = (response.choices[0].message.content or "{}").strip()
            raw = self._extract_json(raw)
            payload = json.loads(raw)

            # Extract dimension scores
            dim_scores: dict[str, float] = {}
            for dim in DIMENSION_WEIGHTS:
                val = payload.get(dim, 0.0)
                dim_scores[dim] = float(val) if isinstance(val, (int, float)) else 0.0
                reasons.append(f"{dim}: {dim_scores[dim]:.2f}")

            if payload.get("reasoning"):
                reasons.append(f"Judge: {payload['reasoning']}")

            return dim_scores

        except Exception as e:
            reasons.append(f"LLM judge failed ({e}), using structural score only")
            # Fallback: neutral scores
            return {dim: 0.5 for dim in DIMENSION_WEIGHTS}

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

    def _check_answer_relevance(self, question: str, answer: str) -> float:
        """Quick heuristic check if answer is relevant to question.
        
        Returns penalty 0.0-0.3 if answer seems off-topic.
        """
        if not question or not answer:
            return 0.0
        
        # Extract key terms from question (words with 4+ chars)
        question_words = set(
            w.lower().strip("?.,!")
            for w in question.split()
            if len(w) >= 4
        )
        
        # Check how many key question terms appear in answer
        answer_lower = answer.lower()
        matches = sum(1 for w in question_words if w in answer_lower)
        
        if not question_words:
            return 0.0
        
        match_rate = matches / len(question_words)
        
        # If less than 30% of key terms appear, apply penalty
        if match_rate < 0.3:
            return 0.3
        elif match_rate < 0.5:
            return 0.1
        
        return 0.0