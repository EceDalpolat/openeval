# openeval/dataset.py
#
# JSONL dataset loader — reads a .jsonl file and turns it into a list of EvalCase.
#
# JSONL = "JSON Lines": each LINE is a separate JSON object. An example line:
#   {"question": "What is RAG?", "answer": "...", "context": "..."}
#
# Why JSONL? Because any project (agentic-rag, finance) can dump its output line by
# line into this file and tell openeval "score this". No more writing Python by hand.

import json
from pathlib import Path

from .judge.schemas import EvalCase


def load_cases(path: str | Path) -> list[EvalCase]:
    """
    Reads a .jsonl file and returns a list of EvalCase.

    Each line must contain these fields:
      - question (required): the question being asked
      - answer   (required): the answer the system gave (openeval scores this)
      - context  (optional): retrieved text, if RAG is used

    On a bad line, it states clearly what went wrong and on which line (it does not skip silently).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    cases: list[EvalCase] = []

    with path.open(encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw or raw.startswith("#"):  # empty line / comment → skip
                continue

            try:
                data = json.loads(raw)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"{path}:{line_no} is not valid JSON → {e.msg}"
                ) from e

            if "question" not in data or "answer" not in data:
                raise ValueError(
                    f"{path}:{line_no} the 'question' and 'answer' fields are required. "
                    f"Fields found: {list(data.keys())}"
                )

            cases.append(
                EvalCase(
                    question=data["question"],
                    answer=data["answer"],
                    context=data.get("context"),
                )
            )

    if not cases:
        raise ValueError(f"No valid cases in {path} (is the file empty?)")

    return cases
