# Stem Agent (LLM + Tooling setup)

Minimal setup so your current agent runs through OpenAI API and can use modular tools.

## 1) Setup environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install openai python-dotenv
```

## 2) Configure API key

Create `.env` in project root (you can copy `.env.example`):

```bash
cp .env.example .env
```

Then edit `.env` and set:

```env
OPENAI_API_KEY=your_real_key_here
SERPAPI_API_KEY=your_serpapi_key_here
```

`SERPAPI_API_KEY` is needed only for web search tool.

## 3) Run

```bash
python main.py
```

You will be prompted for:
- task class (`Deep Research`, `QA`, `Security`)
- mode: `execute`, `specialize`, `evolve`, or `benchmark`
- question (for all modes except `benchmark`)

After `specialize` or `evolve`, app enters a multi-question chat loop.
Commands inside chat loop:
- `/back` -> go to class/mode selection
- `/exit` -> exit program

The agent now calls OpenAI with model from `agents/agent_config.json`.

## Specialization v1 flow

Implemented simple loop in `specialization/`:

- `explorer.py`: researches internet using available tools and proposes workflow
- `architect.py`: converts findings into pipeline + skills/tools plan
- `builder.py`: writes skill markdown files into `skills/` and updates `agents/agent_config.json`
- `evaluator.py`: computes stop score (heuristic v1)
- `pipeline.py`: orchestrates the full flow

To run specialization, choose mode `specialize` in `main.py` prompt.

## Evolution (multi-iteration + rollback)

Mode: `evolve`

What it does:

1. Runs specialization loop for up to `max_iterations` (currently 3)
2. Evaluates each iteration with:
   - structural evaluator score
   - benchmark score
3. Computes combined score and tracks best iteration
4. Rolls back to best discovered `agents/agent_config.json`
5. Saves full report to `eval/results/evolution_<task_class>_<timestamp>.json`

This gives you measurable before/after and a reproducible evolution trail.

## Session isolation and cleanup

- Each specialization/evolution run creates an isolated session under `runs/<session_id>/`
- Specialized config and generated skill markdown files are stored only there
- On `/back` or `/exit`, session directory is deleted automatically
- `agents/agent_config.json` is reset to `agents/agent_config.stem.json` on exit

This ensures every task class starts from a fresh stem agent and no specialization leaks across classes.

## Benchmark suite

Mode: `benchmark`

Benchmark files:

- `eval/benchmark_data.py` — domain-specific cases for Deep Research / QA / Security
- `eval/benchmark_runner.py` — runs baseline vs specialized and computes rubric scores

Scoring (current):

- Baseline answer: plain GPT answer (no specialization)
- Specialized answer: current specialized agent
- Both are judged with a **domain-specific rubric** (1-5 on 5 dimensions, total `/25`):
  - Deep Research: source coverage, factual accuracy, nuance, structure, actionability
  - QA: requirement coverage, test design quality, reproducibility, risk prioritization, actionability
  - Security: threat coverage, technical correctness, prioritization, mitigation quality, evidence/standards

Benchmark reports:

- `baseline_average_total`
- `specialized_average_total`
- `delta_total` (after - before)
- `comparative_win_rate` (pairwise A/B/TIE judge)
- breakdown by domain and by difficulty (`easy`, `medium`, `hard`, `adversarial`)

Use benchmark mode before and after evolution to report measurable gain.

## Built-in tools (`tools/`)

You now have separate tools and a runtime registry:

- `web_search` (SerpAPI)
- `url_fetch` (fetch raw content from URL)
- `arxiv_search` (arXiv API)
- `wikipedia_search` (Wikipedia API)
- `citation_extractor` (extract URLs/DOIs)
- `text_chunker` (split long text)
- `diff_compare` (unified diff)
- `checklist_validator` (required-points QA check)
- `security_pattern_scan` (regex-based security heuristics)
- `file_read`
- `file_write`

Registry file: `tools/tool_library.py`

Tool selection is now **dynamic**: agent decides at runtime which tools to use
based on `task_class` + `question`.

## Runtime extension

You can add new tools during runtime (important for evolution stage):

```python
from tools.base import BaseTool
from tools.tool_library import register_tool

class MyTool(BaseTool):
    name = "my_tool"
    description = "Custom runtime tool"

    def run(self, context: dict[str, object]) -> str:
        return "ok"

register_tool(MyTool())
```

Then include the tool name in your evolving config's `tools` list and reference it from skill definitions.

## Notes

- Current default model: `gpt-4o-mini`
- Config path fixed to `agents/agent_config.json`
- Tool loading is dynamic; if some tool fails to load, agent still starts
