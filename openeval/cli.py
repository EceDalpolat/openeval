# openeval/cli.py
#
# Terminal command. Goal: score answers by running `openeval run cases.jsonl`, no code needed.
#
#   openeval run examples/sample_cases.jsonl
#   openeval run answers.jsonl --judge-model openai/gpt-4o --judge-provider openrouter
#
# openeval does NOT produce answers — it has a "judge" model score the pre-generated answers in the dataset.

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv


def _build_connector(provider: str, model: str):
    """Builds the right connector based on the provider name."""
    if provider == "openrouter":
        from .connectors.openrouter_connector import OpenRouterConnector
        return OpenRouterConnector(model=model)
    if provider == "openai":
        from .connectors.openai_connector import OpenAIConnector
        return OpenAIConnector(model=model)
    if provider == "ollama":
        from .connectors.ollama_connector import OllamaConnector
        return OllamaConnector(model=model)
    raise ValueError(f"Unknown provider: {provider}")


def cmd_run(args: argparse.Namespace) -> int:
    load_dotenv()  # load API keys from .env

    # Do the heavy imports when the command runs (keeps CLI startup fast)
    from .dataset import load_cases
    from .eval.evaluator import Evaluator

    cases = load_cases(args.cases)

    judge = _build_connector(args.judge_provider, args.judge_model)
    if not judge.is_available():
        print(
            f"❌ Judge not available: {args.judge_provider}/{args.judge_model}\n"
            f"   OpenRouter/OpenAI need an API key in .env; Ollama needs the local service running.",
            file=sys.stderr,
        )
        return 1

    # Label for the system that produced the answers: defaults to the dataset filename
    label = args.label or Path(args.cases).stem

    evaluator = Evaluator(
        judge_connector=judge, subject_label=label, dataset=str(args.cases)
    )
    report = evaluator.run(cases)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(f"\n✅ Report saved: {out}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    from rich.console import Console
    from rich.table import Table

    from .compare import diff_reports
    from .judge.schemas import EvalReport

    before = EvalReport.model_validate_json(
        Path(args.before).read_text(encoding="utf-8")
    )
    after = EvalReport.model_validate_json(
        Path(args.after).read_text(encoding="utf-8")
    )

    label_b = args.label_before or Path(args.before).stem
    label_a = args.label_after or Path(args.after).stem

    console = Console()
    table = Table(title=f"Comparison — {label_b} → {label_a}")
    table.add_column("Dimension", style="cyan")
    table.add_column(label_b, justify="right")
    table.add_column(label_a, justify="right")
    table.add_column("Δ", justify="right")

    for row in diff_reports(before, after):
        d = row["delta"]
        color = "green" if d > 0 else "red" if d < 0 else "dim"
        sign = "+" if d > 0 else ""
        style = "bold" if row["dimension"] == "overall" else ""
        table.add_row(
            f"[{style}]{row['dimension']}[/{style}]" if style else row["dimension"],
            f"{row['before']:.2f}",
            f"{row['after']:.2f}",
            f"[{color}]{sign}{d:.2f}[/{color}]",
        )
    console.print(table)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="openeval",
        description="Lightweight LLM-as-judge evaluation — score answers with a judge model.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Score a JSONL dataset")
    p_run.add_argument("cases", help="The .jsonl dataset to score")
    p_run.add_argument(
        "--judge-model",
        default="meta-llama/llama-3.2-3b-instruct:free",
        help="Judge model (default: OpenRouter free Llama 3.2 3B)",
    )
    p_run.add_argument(
        "--judge-provider",
        default="openrouter",
        choices=["openrouter", "openai", "ollama"],
        help="Judge provider (default: openrouter)",
    )
    p_run.add_argument(
        "--out",
        default="reports/report.json",
        help="Report output path (default: reports/report.json)",
    )
    p_run.add_argument(
        "--label",
        default=None,
        help="Name of the system that produced the answers (default: dataset filename)",
    )
    p_run.set_defaults(func=cmd_run)

    p_cmp = sub.add_parser("compare", help="Compare two reports (before → after)")
    p_cmp.add_argument("before", help="Earlier report (JSON)")
    p_cmp.add_argument("after", help="Later report (JSON)")
    p_cmp.add_argument("--label-before", default=None, help="Label for the 'before' column")
    p_cmp.add_argument("--label-after", default=None, help="Label for the 'after' column")
    p_cmp.set_defaults(func=cmd_compare)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
