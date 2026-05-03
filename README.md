# Stem Agent

A self-specializing AI agent that grows into a domain expert through its own process.

## Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install openai python-dotenv

# Create .env file
echo "OPENAI_API_KEY=your_key_here" > .env
echo "SERPAPI_API_KEY=your_key_here" > .env
```

## Run

```bash

python main.py
```

### Usage

1. Enter task class: `Deep Research`, `QA`, or `Security`
2. Choose mode: `execute` or `specialize`
3. For specialize: enter a question

Example:
```
Enter task class: Deep Research
Mode: specialize
Enter question: what is RAG
```

### Commands (in execute mode)

- `/benchmark` - Run baseline vs specialized comparison
- **IMPORTANT: BENCHMARK IS AUTOMATICALLY BEING RUN DURING SPECIALIZATION SO THERE IS NOW NEED TO RUN IT AGAIN**
- `/back` - Return to menu
- `/exit` - Exit

## How it works

```
Task Class → Explorer (senses environment)
          → Architect (decides architecture)
          → Builder (builds specialized config)
          → Evaluator (checks if ready)
          → Auto-learn (improves from failures)
```

## Output

- `eval/results/comparison_*.json` - Baseline vs specialized comparison
- `skills/learnings.md` - Learned improvements (auto-generated)

## Requirements

- Python 3.11+
- OpenAI API key
