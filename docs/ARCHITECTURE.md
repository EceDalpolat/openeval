# openeval — Architecture

openeval is a lightweight evaluation tool that measures one LLM's answers by having
another LLM (the "judge") grade them. The method is known in the literature as
**LLM-as-judge**.

**Why?** Saying "my system gives good answers" is not enough — you have to measure it.
Instead of reading thousands of answers by hand, we make a strong model the judge and
have it grade every answer on 5 dimensions from 0 to 1.

---

## Data flow

```
dataset.jsonl                          [1] dataset.py: load_cases()
  │  {question, answer, context}
  ▼
list[EvalCase]                         [2] cli.py: set up the judge connector (ollama/openai/…)
  │
  ▼
Evaluator.run(cases)                   [3] eval/evaluator.py: orchestrator
  │  for each case ────────────┐
  ▼                            │
Judge.evaluate(case)           │       [4] judge/judge.py: the actual grader
  │  build prompt → send to    │
  │  the judge model → parse   │
  │  the JSON                  │
  ▼                            │
EvaluationResult (5 dims)   ◄──┘       [5] judge/schemas.py: scores + overall
  │
  ▼  (accumulate + average)
EvalReport  ──► reports/report.json    [6] metadata + averages + cost
  │
  ▼
compare a.json b.json                  [7] compare.py: before/after Δ
```

---

## Components

### Schemas — `judge/schemas.py`
The system's data types:

- **`EvalCase`** — a single test: `question`, `answer`, `context` (optional RAG text).
- **`DimensionScore`** — a single dimension's grade: `score` (0.0–1.0) + `reasoning`.
- **`EvaluationResult`** — a 5-dimension grade for one answer. `overall_score` is a
  **weighted** average (see the table below). If an answer is fluent but wrong, overall
  still drops.
- **`EvalReport`** — the run's summary: dimension averages + tokens/cost/latency +
  metadata (`judge_model`, `dataset`, `created_at`).

### The 5 dimensions
| Dimension | Measures | Weight |
|---|---|---|
| faithfulness | Is the answer correct? | 0.30 |
| relevance | Does it answer the question? | 0.30 |
| clarity | Is it understandable? | 0.20 |
| safety | Is it harmful/unethical? | 0.10 |
| consistency | Does it contradict itself? | 0.10 |

### Connectors — `connectors/`
The layer that talks to the models. `BaseConnector` is a contract (abstract): every
connector provides `generate(prompt)`, `is_available()`, `model_name`. Implementations:
OpenAI, OpenRouter, Ollama. Thanks to the shared interface, switching the judge model is
a one-liner.

**subject vs judge — a key distinction:**
- **subject**: the system that *produces* the answers (e.g. agentic-rag). openeval does
  **not run** it, it only keeps its name as a label (`subject_label`).
- **judge**: the model that *grades* the answers — usually chosen to be stronger than the subject.

Because openeval scores pre-generated answers, the subject connector is not required;
the judge alone is enough.

### The judge — `judge/judge.py`
Steps in `evaluate(case)`:
1. Builds a prompt (question + answer + context) with the instruction "return ONLY this JSON".
2. Sends it to the judge model with `temperature=0` → reproducibility.
3. Parses the response robustly:
   - `extract_json()` — pulls out the JSON even if the model adds a ```` ```fence ```` or a preamble.
   - `coerce_dimension()` — missing/malformed dimension → neutral 0.0 + note; `90→0.9`; clamped to 0–1.
   - retry — retries `MAX_RETRIES` times on a transient API error.
4. Returns an `EvaluationResult`. A single bad answer cannot crash the whole run.

### Orchestrator — `eval/evaluator.py`
Feeds cases to the judge one by one, collects the results, computes the averages, adds the
metadata and metrics, produces an `EvalReport`, and prints the summary table. This is the
main class the user interacts with.

### Metrics — `observability/metrics.py`
- `Timer` — call duration.
- `CallMetrics` — a single call's tokens/cost/latency. `cost_usd` is priced by the model
  name; `ollama/…` and `local/…` models are $0 (local, free).
- `SessionMetrics` — the total for the whole run.

### Loader / Compare / CLI
- `dataset.py` — JSONL → `list[EvalCase]`; gives a clear error on a bad line.
- `compare.py` — `diff_reports(before, after)` computes the difference dimension by dimension.
- `cli.py` — the `run` (score a dataset) and `compare` (compare two reports) commands.

---

## Example: the story of a single answer
Input: `"What is RAG?"` + `"It's a thing."` →
It is embedded in the judge prompt → goes to the model → the judge returns:
`faithfulness 0.2, relevance 0.3, clarity 0.3, safety 1.0, consistency 0.6` →
overall = 0.2·0.3 + 0.3·0.3 + 0.3·0.2 + 1.0·0.1 + 0.6·0.1 ≈ **0.31** (low — an empty answer).

---

## Usage

```bash
# Score a dataset (with a local Ollama judge)
openeval run examples/sample_cases.jsonl --judge-provider ollama --judge-model llama3.2

# Compare two runs (before → after)
openeval compare reports/report_before.json reports/report.json
```

---

## An honest limitation — "the judge is also an LLM, why trust it?"
We don't fully trust it. The judge can be biased and can be generous on easy questions.
Ways to mitigate: (a) pick a judge stronger than the subject, (b) make it deterministic
with `temperature=0`, (c) keep the `reasoning` so every grade is auditable, (d) multiple
judges / human spot-checks (beyond this tool's current scope). Knowing this limitation is
part of understanding the system.
