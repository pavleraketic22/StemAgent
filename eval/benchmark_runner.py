from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from agents.agent import Agent
from eval.benchmark_data import BENCHMARK_CASES, BenchmarkCase


DEEP_RESEARCH_RUBRIC = """
You are evaluating a DEEP RESEARCH response. Score each dimension 1-5.

1. SOURCE_COVERAGE: 1=no sources, 5=multiple diverse high-quality sources
2. FACTUAL_ACCURACY: 1=contains errors/hallucinations, 5=claims are accurate and verifiable
3. NUANCE: 1=overconfident, 5=clear known/debated/unknown distinction
4. STRUCTURE: 1=list of facts, 5=coherent synthesis
5. ACTIONABILITY: 1=not useful, 5=clear takeaway/next steps

Return ONLY valid JSON:
{"source_coverage": X, "factual_accuracy": X, "nuance": X, "structure": X, "actionability": X, "reasoning": "one sentence"}
""".strip()


QA_RUBRIC = """
You are evaluating a QA response. Score each dimension 1-5.

1. REQUIREMENT_COVERAGE: 1=major gaps, 5=comprehensive coverage of expected test scope
2. TEST_DESIGN_QUALITY: 1=vague tests, 5=clear positive/negative/edge-case design
3. REPRODUCIBILITY: 1=not executable, 5=steps and checks are concrete and reproducible
4. RISK_PRIORITIZATION: 1=no prioritization, 5=clear severity/impact-based priorities
5. ACTIONABILITY: 1=not useful, 5=immediately actionable QA plan/checklist

Return ONLY valid JSON:
{"requirement_coverage": X, "test_design_quality": X, "reproducibility": X, "risk_prioritization": X, "actionability": X, "reasoning": "one sentence"}
""".strip()


SECURITY_RUBRIC = """
You are evaluating a SECURITY response. Score each dimension 1-5.

1. THREAT_COVERAGE: 1=misses key threats, 5=covers major realistic attack paths
2. TECHNICAL_CORRECTNESS: 1=unsafe/incorrect advice, 5=technically correct and safe guidance
3. PRIORITIZATION: 1=no prioritization, 5=clear severity and risk-driven prioritization
4. MITIGATION_QUALITY: 1=generic advice, 5=specific layered mitigations and controls
5. EVIDENCE_AND_STANDARDS: 1=no references, 5=uses credible references/standards (OWASP, CWE, NIST, etc.)

Return ONLY valid JSON:
{"threat_coverage": X, "technical_correctness": X, "prioritization": X, "mitigation_quality": X, "evidence_and_standards": X, "reasoning": "one sentence"}
""".strip()


@dataclass
class CaseResult:
    domain: str
    difficulty: str
    question: str
    baseline_total: float
    specialized_total: float
    delta_total: float
    comparative_label: str
    comparative_win: bool
    baseline_scores: dict[str, Any]
    specialized_scores: dict[str, Any]


@dataclass
class BenchmarkResult:
    baseline_average_total: float
    specialized_average_total: float
    delta_total: float
    comparative_win_rate: float
    baseline_by_difficulty: dict[str, float]
    specialized_by_difficulty: dict[str, float]
    baseline_by_domain: dict[str, float]
    specialized_by_domain: dict[str, float]
    case_results: list[CaseResult]


class BenchmarkRunner:
    def __init__(
        self,
        config_path: str = "agents/agent_config.json",
        baseline_config_path: str = "agents/agent_config.stem.json",
        answer_model: str = "gpt-4o-mini",
        judge_model: str = "gpt-4o",
    ) -> None:
        load_dotenv()
        self.config_path = config_path
        self.baseline_config_path = baseline_config_path
        self.answer_model = answer_model
        self.judge_model = judge_model

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for benchmark execution.")
        self.client = OpenAI(api_key=api_key)

    def run(self, cases: list[BenchmarkCase] | None = None) -> BenchmarkResult:
        active_cases = cases or BENCHMARK_CASES
        specialized_agent = Agent(self.config_path)
        baseline_agent = Agent(self.baseline_config_path)
        results: list[CaseResult] = []

        for case in active_cases:
            baseline_answer = self._run_baseline_answer(case.question)
            specialized_response = specialized_agent.run(question=case.question, task_class=case.domain)
            specialized_answer = str(specialized_response.get("answer", ""))

            baseline_scores = self._judge_response(
                case=case,
                response_text=baseline_answer,
            )
            specialized_scores = self._judge_response(
                case=case,
                response_text=specialized_answer,
            )

            comparative_label = self._comparative_judge(
                question=case.question,
                baseline_answer=baseline_answer,
                specialized_answer=specialized_answer,
                domain=case.domain,
            )
            comparative_win = comparative_label == "SPECIALIZED"

            baseline_total = float(baseline_scores["total"])
            specialized_total = float(specialized_scores["total"])

            results.append(
                CaseResult(
                    domain=case.domain,
                    difficulty=case.difficulty,
                    question=case.question,
                    baseline_total=baseline_total,
                    specialized_total=specialized_total,
                    delta_total=specialized_total - baseline_total,
                    comparative_label=comparative_label,
                    comparative_win=comparative_win,
                    baseline_scores=baseline_scores,
                    specialized_scores=specialized_scores,
                )
            )

        baseline_avg = mean([r.baseline_total for r in results]) if results else 0.0
        specialized_avg = mean([r.specialized_total for r in results]) if results else 0.0
        win_rate = mean([1.0 if r.comparative_win else 0.0 for r in results]) if results else 0.0

        baseline_by_difficulty = self._aggregate_by(results, key="difficulty", value="baseline_total")
        specialized_by_difficulty = self._aggregate_by(
            results, key="difficulty", value="specialized_total"
        )
        baseline_by_domain = self._aggregate_by(results, key="domain", value="baseline_total")
        specialized_by_domain = self._aggregate_by(results, key="domain", value="specialized_total")

        return BenchmarkResult(
            baseline_average_total=baseline_avg,
            specialized_average_total=specialized_avg,
            delta_total=specialized_avg - baseline_avg,
            comparative_win_rate=win_rate,
            baseline_by_difficulty=baseline_by_difficulty,
            specialized_by_difficulty=specialized_by_difficulty,
            baseline_by_domain=baseline_by_domain,
            specialized_by_domain=specialized_by_domain,
            case_results=results,
        )

    def _run_baseline_answer(self, question: str) -> str:
        response = self.client.chat.completions.create(
            model=self.answer_model,
            temperature=0,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question},
            ],
        )
        return response.choices[0].message.content or ""

    def _judge_response(self, *, case: BenchmarkCase, response_text: str) -> dict[str, Any]:
        rubric, expected_keys = self._rubric_for_domain(case.domain)
        citation_hint = "Citations are expected." if case.expect_citations else "Citations are optional."

        prompt = (
            "Domain: {domain}\n"
            "Difficulty: {difficulty}\n"
            "Question asked:\n{question}\n\n"
            "Response to evaluate:\n{response_text}\n\n"
            "{citation_hint}\n\n"
            "{rubric}"
        ).format(
            domain=case.domain,
            difficulty=case.difficulty,
            question=case.question,
            response_text=response_text,
            citation_hint=citation_hint,
            rubric=rubric,
        )

        result = self.client.chat.completions.create(
            model=self.judge_model,
            temperature=0,
            messages=[
                {"role": "system", "content": "Return strict JSON only."},
                {"role": "user", "content": prompt},
            ],
        )

        raw = (result.choices[0].message.content or "{}").strip()
        parsed = json.loads(self._extract_json_text(raw))

        normalized: dict[str, Any] = {"reasoning": str(parsed.get("reasoning", ""))}
        total = 0
        for key in expected_keys:
            score = self._bounded_int(parsed.get(key))
            normalized[key] = score
            total += score

        normalized["total"] = total
        return normalized

    def _comparative_judge(
        self,
        *,
        question: str,
        baseline_answer: str,
        specialized_answer: str,
        domain: str,
    ) -> str:
        if random.random() < 0.5:
            response_a = baseline_answer
            response_b = specialized_answer
            specialized_label = "B"
        else:
            response_a = specialized_answer
            response_b = baseline_answer
            specialized_label = "A"

        prompt = (
            "Domain: {domain}\n"
            "Question: {question}\n\n"
            "Response A:\n{a}\n\n"
            "Response B:\n{b}\n\n"
            "Which response is better overall for this domain?\n"
            "Judge on correctness, completeness, and practical usefulness.\n"
            "Return only one token: A, B, or TIE."
        ).format(domain=domain, question=question, a=response_a, b=response_b)

        result = self.client.chat.completions.create(
            model=self.judge_model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict evaluator. Return exactly A, B, or TIE.",
                },
                {"role": "user", "content": prompt},
            ],
        )

        raw = (result.choices[0].message.content or "TIE").strip().upper()
        label = raw.split()[0] if raw else "TIE"
        if label not in {"A", "B", "TIE"}:
            label = "TIE"

        if label == "TIE":
            return "TIE"
        return "SPECIALIZED" if label == specialized_label else "BASELINE"

    @staticmethod
    def _rubric_for_domain(domain: str) -> tuple[str, tuple[str, str, str, str, str]]:
        normalized = domain.strip().lower()
        if normalized == "qa":
            return QA_RUBRIC, (
                "requirement_coverage",
                "test_design_quality",
                "reproducibility",
                "risk_prioritization",
                "actionability",
            )
        if normalized == "security":
            return SECURITY_RUBRIC, (
                "threat_coverage",
                "technical_correctness",
                "prioritization",
                "mitigation_quality",
                "evidence_and_standards",
            )
        return DEEP_RESEARCH_RUBRIC, (
            "source_coverage",
            "factual_accuracy",
            "nuance",
            "structure",
            "actionability",
        )

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

    @staticmethod
    def _bounded_int(value: Any, low: int = 1, high: int = 5) -> int:
        try:
            numeric = int(value)
        except (ValueError, TypeError):
            return low
        return max(low, min(high, numeric))

    @staticmethod
    def _aggregate_by(results: list[CaseResult], *, key: str, value: str) -> dict[str, float]:
        grouped: dict[str, list[float]] = {}
        for item in results:
            group_key = str(getattr(item, key))
            grouped.setdefault(group_key, []).append(float(getattr(item, value)))
        return {k: mean(v) for k, v in grouped.items()}


def to_dict(result: BenchmarkResult) -> dict[str, Any]:
    return {
        "baseline_average_total": result.baseline_average_total,
        "specialized_average_total": result.specialized_average_total,
        "delta_total": result.delta_total,
        "comparative_win_rate": result.comparative_win_rate,
        "baseline_by_difficulty": result.baseline_by_difficulty,
        "specialized_by_difficulty": result.specialized_by_difficulty,
        "baseline_by_domain": result.baseline_by_domain,
        "specialized_by_domain": result.specialized_by_domain,
        "case_results": [asdict(case) for case in result.case_results],
    }
