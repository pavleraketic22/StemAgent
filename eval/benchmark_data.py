from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkCase:
    domain: str
    difficulty: str
    question: str
    expect_citations: bool = True
    required_keywords: tuple[str, ...] = ()


DEEP_RESEARCH_BY_DIFFICULTY: dict[str, list[str]] = {
    "easy": [
        "What are the main differences between RAG and fine-tuning for LLM adaptation?",
        "What does the research say about sleep deprivation and cognitive performance?",
    ],
    "medium": [
        "What are the failure modes of RLHF and what alternatives have been proposed?",
        "How do EU AI Act requirements differ from US executive orders on AI governance?",
    ],
    "hard": [
        "What does empirical research say about the effectiveness of code review practices?",
        "How have AI coding assistants measurably changed developer productivity — what does the evidence actually show?",
    ],
    "adversarial": [
        "Is intermittent fasting effective for weight loss? (conflicting evidence domain)",
    ],
}


QA_BY_DIFFICULTY: dict[str, list[str]] = {
    "easy": [
        "Create a QA checklist for reviewing a REST API endpoint.",
        "How would you validate that pagination works correctly in an API?",
    ],
    "medium": [
        "Design a test plan for a file upload feature (size, format, failures).",
        "What test strategy would you use for feature flags across environments?",
    ],
    "hard": [
        "How would you test eventual consistency in a distributed order processing workflow?",
        "Propose a QA approach for zero-downtime database migration validation.",
    ],
    "adversarial": [
        "A stakeholder asks to skip regression before release due to deadline. What is your QA response?",
    ],
}


SECURITY_BY_DIFFICULTY: dict[str, list[str]] = {
    "easy": [
        "How do you prevent SQL injection in Python web applications?",
        "What are common signs of hardcoded secrets in repositories?",
    ],
    "medium": [
        "How would you secure subprocess execution in backend services?",
        "What defenses mitigate SSRF in cloud-native applications?",
    ],
    "hard": [
        "Create a threat model for a document-upload SaaS product handling sensitive files.",
        "How would you reduce blast radius after a token leakage incident?",
    ],
    "adversarial": [
        "A developer says 'we'll sanitize inputs later' before launch. What concrete security gates do you enforce now?",
    ],
}


# Domain-specific keywords for objective scoring
DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "Deep Research": ("according to", "study", "research", "source", "data", "evidence", "paper", "journal"),
    "QA": ("test", "case", "validate", "coverage", "reproduce", "assertion", "scenario", "edge"),
    "Security": ("vulnerability", "threat", "mitigation", "control", "secure", "risk", "exploit", "owasp"),
}


def _build_cases(
    *,
    domain: str,
    by_difficulty: dict[str, list[str]],
    expect_citations: bool,
    required_keywords: tuple[str, ...] | None = None,
) -> list[BenchmarkCase]:
    keywords = required_keywords or DOMAIN_KEYWORDS.get(domain, ())
    return [
        BenchmarkCase(
            domain=domain,
            difficulty=difficulty,
            question=question,
            expect_citations=expect_citations,
            required_keywords=keywords,
        )
        for difficulty, questions in by_difficulty.items()
        for question in questions
    ]


BENCHMARK_CASES: list[BenchmarkCase] = (
    _build_cases(
        domain="Deep Research",
        by_difficulty=DEEP_RESEARCH_BY_DIFFICULTY,
        expect_citations=True,
    )
    + _build_cases(
        domain="QA",
        by_difficulty=QA_BY_DIFFICULTY,
        expect_citations=False,
    )
    + _build_cases(
        domain="Security",
        by_difficulty=SECURITY_BY_DIFFICULTY,
        expect_citations=True,
    )
)

# Export keywords for use in evaluator
__all__ = ["BenchmarkCase", "BENCHMARK_CASES", "DOMAIN_KEYWORDS"]
