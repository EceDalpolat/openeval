# OpenEval

[![PyPI](https://img.shields.io/pypi/v/openeval-llm.svg)](https://pypi.org/project/openeval-llm/)
[![Python](https://img.shields.io/pypi/pyversions/openeval-llm.svg)](https://pypi.org/project/openeval-llm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A lightweight **LLM-as-judge** evaluation tool. It takes a dataset of
question/answer pairs, has a strong "judge" model score each answer across 5
dimensions, and compares two runs to show the **before/after** delta.
Works with OpenAI, OpenRouter, or a local Ollama model.

## Install

```bash
pip install openeval-llm
```

> The distribution name is `openeval-llm`; the import and command name is `openeval`.

## Quickstart (CLI)

Score a JSONL dataset — one `{"question", "answer", "context"}` object per line:

```bash
# With a local Ollama judge (free, private)
openeval run cases.jsonl --judge-provider ollama --judge-model llama3.2

# ...or OpenAI / OpenRouter (put the API key in .env)
openeval run cases.jsonl --judge-provider openai --judge-model gpt-4o-mini
```

Output — per-dimension averages, token/cost/latency, and provenance
(judge, dataset, timestamp):

```
       Results — cases
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Dimension     ┃ Avg Score     ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ faithfulness  │ 0.73          │
│ relevance     │ 0.93          │
│ overall       │ 0.85          │
└───────────────┴───────────────┘
```

## Before / after comparison

Run the same system two ways (e.g. a feature on vs off), then compare:

```bash
openeval compare reports/report_before.json reports/report_after.json
```

```
Dimension     before  after    Δ
faithfulness  0.71    0.88   +0.17
overall       0.70    0.85   +0.15
```

## The 5 dimensions

| Dimension | Measures | Weight |
|---|---|---|
| faithfulness | Is the answer factually correct? | 0.30 |
| relevance | Does it address the question? | 0.30 |
| clarity | Is it clear and well-explained? | 0.20 |
| safety | Is it safe / non-harmful? | 0.10 |
| consistency | Is it internally consistent? | 0.10 |

`overall` is the weighted average — faithfulness and relevance dominate.

## Python API

```python
from openeval.connectors.ollama_connector import OllamaConnector
from openeval.dataset import load_cases
from openeval.eval.evaluator import Evaluator

cases = load_cases("cases.jsonl")
evaluator = Evaluator(judge_connector=OllamaConnector(model="llama3.2"))
report = evaluator.run(cases)
print(report.avg_overall)
```

## Highlights

- **Crash-proof judge:** survives when the judge wraps its answer in
  ` ```json ` fences, adds preamble, or omits a dimension (JSON extraction +
  neutral defaults + retry on transient errors).
- **Judge ≠ subject:** pick a judge stronger than the system that produced the
  answers; OpenEval scores pre-generated answers, so it never runs the subject.
- **Provenance in every report:** judge model, dataset, and timestamp are
  recorded for reproducibility.
- **Local = free:** Ollama models are billed at $0.

## Ollama setup

```bash
brew install ollama      # macOS
ollama pull llama3.2
ollama serve
```

## Dashboard (optional)

```bash
streamlit run openeval/report/dashboard.py
```

## Project layout

```text
openeval/
├── connectors/    # model providers (OpenAI / OpenRouter / Ollama)
├── dataset.py     # JSONL loader
├── eval/          # main evaluation flow (Evaluator)
├── judge/         # the scoring LLM-judge logic
├── compare.py     # before/after comparison
├── cli.py         # `openeval run` / `openeval compare`
├── observability/ # logging, token/cost/latency, optional Langfuse
└── report/        # report helpers + Streamlit dashboard
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for a deeper walkthrough.

## License

MIT — see [LICENSE](LICENSE).
