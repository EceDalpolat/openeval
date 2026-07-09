# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

OpenEval is a lightweight LLM-as-judge evaluation framework. It runs question/answer cases through a "judge" model that scores each answer on 5 dimensions, aggregates the results into a report, and surfaces them in a Streamlit dashboard. Optional RAG retrieval injects context into cases, and an observability layer tracks tokens/cost/latency. The codebase, comments, and docstrings are written in Turkish.

## Commands

```bash
pip install -e .            # install the package
pip install -e ".[dev]"     # + pytest, pytest-asyncio, ruff, mypy

ruff check .                # lint (line-length 88)
mypy openeval               # type check
pytest                      # run tests
pytest tests/test_judge.py::test_name   # single test

python examples/basic_eval.py                 # run an evaluation → writes reports/eval_report.json
streamlit run openeval/report/dashboard.py    # view the latest report
```

The dashboard reads `reports/eval_report.json`, so an evaluation must be run before the dashboard shows anything.

### External services
- **Ollama** must be running (`ollama serve`) for the local judge and for RAG. Pull both models: `ollama pull llama3.2:3b` (judge) and `ollama pull nomic-embed-text` (embeddings). The embedder/retriever hit `http://localhost:11434`.
- **OpenRouter / OpenAI** connectors need `OPENROUTER_API_KEY` / `OPENAI_API_KEY` in `.env` (loaded via `python-dotenv`).
- **Langfuse** tracing is optional and silently disabled unless the `langfuse` package is installed *and* `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` are set.

## Architecture

The data flows in one direction through these layers:

**Connectors** (`openeval/connectors/`) — All implement `BaseConnector` (abstract: `generate()`, `is_available()`, `model_name`) and return a uniform `ModelResponse(content, model, input_tokens, output_tokens)`. This is the swap point: `OpenAIConnector`, `OllamaConnector`, and `OpenRouterConnector` are interchangeable anywhere a connector is expected. All call `generate` with `temperature=0` for deterministic eval.

**Judge** (`openeval/judge/`) — `Judge.evaluate(case)` builds a prompt from `JUDGE_SYSTEM` + `JUDGE_PROMPT`, sends it through a connector, and parses the response as JSON into an `EvaluationResult`. The judge demands a strict JSON object with the 5 dimensions; the parser strips markdown code fences and normalizes scores >1 (e.g. model returns `90` → `0.9`). `schemas.py` defines all Pydantic models (`EvalCase`, `EvaluationResult`, `DimensionScore`, `EvalReport`).

**Evaluator** (`openeval/eval/evaluator.py`) — The single public entry point. `Evaluator(connector, judge_connector=None)` lets you use a separate (stronger) model as judge; if omitted the same connector judges itself. `run(cases)` loops cases through the judge, accumulates `SessionMetrics`, prints a Rich summary table, and returns an `EvalReport` with per-dimension averages plus token/cost/latency totals.

**RAG** (`openeval/rag/`) — `ChromaRetriever` indexes the hardcoded `DOCUMENTS` in `knowledge_base.py` into a persistent ChromaDB store (`.chromadb/`), embedding each chunk via `OllamaEmbedder` (nomic-embed-text, 768-dim, cosine). `retrieve_as_context(query)` returns a context string to attach to an `EvalCase`. See `examples/basic_eval.py` for the `make_case` pattern that auto-attaches retrieved context.

**Observability** (`openeval/observability/`) — `get_logger()` is a central factory (Rich console at INFO, daily file at `logs/openeval_YYYY-MM-DD.log` at DEBUG). `SessionMetrics`/`CallMetrics`/`Timer` track and aggregate per-call cost & latency. `tracer` is a module-level singleton wrapping Langfuse. These are threaded through connectors, the judge, and the evaluator.

### The 5 scoring dimensions
`faithfulness`, `relevance`, `clarity`, `safety`, `consistency` — each 0.0–1.0. `overall_score` is a weighted average defined in `EvaluationResult.overall_score`: faithfulness 0.30, relevance 0.30, clarity 0.20, safety 0.10, consistency 0.10.

## Gotchas

- **The `overall_score` weights are duplicated.** They live in `schemas.py:EvaluationResult.overall_score` *and* are re-hardcoded in `dashboard.py`. Change both together.
- **`dashboard.py` has a hardcoded `questions` list** that it `zip()`s against results to label cases. It is currently out of sync with `examples/basic_eval.py` (10 labels vs 8 cases, different questions), so dashboard case labels are wrong. Keep them in sync, or better, persist the questions in the report instead of hardcoding.
- **`OpenRouterConnector` skips observability** — unlike the OpenAI/Ollama connectors it doesn't use `Timer`, `CallMetrics`, or `tracer`, so its calls won't appear in metrics/traces.
- **Cost numbers are only accurate for `gpt-4o`/`gpt-4o-mini`.** `COST_PER_1K_TOKENS` in `metrics.py` falls back to gpt-4o-mini pricing for any other model (Ollama, OpenRouter), so reported `cost_usd` for those is fictional.
- **ChromaDB is cached.** `ChromaRetriever` reuses the existing collection across runs; after editing `knowledge_base.py` call `retriever.reset()` (or delete `.chromadb/`) to re-index.
- **The judge raises on malformed JSON.** A model that emits non-JSON (common with small local models) will crash `Judge.evaluate` with a `JSONDecodeError`.

## Status / incomplete pieces

- `tests/test_judge.py` is empty — there are no actual tests yet despite the pytest dev dependency.
- `openeval/report/reporter.py` and `report/__init__.py` are empty stubs; `dashboard.py` is the only working reporting surface.
- `examples/create_finetune_dataset.py` writes a small static JSONL to `data/`; fine-tuning itself is not implemented.
