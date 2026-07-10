# openeval/eval/evaluator.py

from datetime import datetime, timezone

from rich.console import Console
from rich.table import Table

from ..connectors.base import BaseConnector
from ..judge.judge import Judge
from ..judge.schemas import EvalCase, EvalReport
from ..observability import SessionMetrics, get_logger, tracer

console = Console()

class Evaluator:
    """
    The main class. This is the only thing the user interacts with.

    Example:
        evaluator = Evaluator(connector=OpenAIConnector())
        report = evaluator.run(cases)
    """

    def __init__(
        self,
        connector: BaseConnector | None = None,
        judge_connector: BaseConnector | None = None,
        tracer_client: object | None = None,
        subject_label: str | None = None,
        dataset: str | None = None,
    ):
        # openeval does NOT produce answers — it SCORES the pre-generated answers in the dataset.
        # So the subject connector is optional; the judge alone is enough.
        if connector is None and judge_connector is None:
            raise ValueError(
                "At least one connector is required (judge_connector or connector)."
            )
        self.connector = connector
        self.logger = get_logger(__name__)
        self.metrics = SessionMetrics()
        self.tracer = tracer_client or tracer
        # You can pass a separate (stronger) model for the judge; if not, the subject connector is used.
        self.judge = Judge(
            judge_connector or connector,
            metrics=self.metrics,
            tracer_client=self.tracer,
        )
        # Label of the system that produced the answers (for "which system did we measure" in the report).
        self.subject_label = subject_label or (
            connector.model_name if connector else "pre-generated"
        )
        self.dataset = dataset

    def run(self, cases: list[EvalCase]) -> EvalReport:
        results = []

        self.logger.info("Evaluation started: cases=%d, model=%s", len(cases), self.subject_label)
        if getattr(self.tracer, "start_trace", None):
            self.tracer.start_trace(
                name="openeval.run",
                metadata={
                    "model": self.subject_label,
                    "judge_model": self.judge.connector.model_name,
                    "total_cases": len(cases),
                },
            )

        console.print(f"\n[bold]OpenEval[/bold] — evaluating {len(cases)} cases\n")

        for i, case in enumerate(cases, 1):
            console.print(f"[{i}/{len(cases)}] {case.question[:60]}...")
            self.logger.info("Processing case %d/%d", i, len(cases))

            # 1. Get an answer from the model (if no connector was given, case.answer is used)
            result = self.judge.evaluate(case)
            results.append(result)
            console.print(f"  → overall: [green]{result.overall_score:.2f}[/green]")
            self.logger.info("Case %d completed: overall=%.2f", i, result.overall_score)

        # Averages
        avg = lambda dim: sum(getattr(r, dim).score for r in results) / len(results)

        report = EvalReport(
            model=self.subject_label,
            judge_model=self.judge.connector.model_name,
            dataset=self.dataset,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            total_cases=len(cases),
            results=results,
            avg_overall=sum(r.overall_score for r in results) / len(results),
            avg_faithfulness=avg("faithfulness"),
            avg_relevance=avg("relevance"),
            avg_clarity=avg("clarity"),
            avg_safety=avg("safety"),
            avg_consistency=avg("consistency"),
            total_input_tokens=self.metrics.total_input_tokens,
            total_output_tokens=self.metrics.total_output_tokens,
            total_tokens=self.metrics.total_tokens,
            total_cost_usd=self.metrics.total_cost_usd,
            avg_latency_ms=self.metrics.avg_latency_ms,
        )

        self._print_summary(report)
        if getattr(self.tracer, "end_trace", None):
            self.tracer.end_trace(
                metadata={
                    "total_cases": report.total_cases,
                    "avg_overall": report.avg_overall,
                    "total_tokens": report.total_tokens,
                    "total_cost_usd": report.total_cost_usd,
                    "avg_latency_ms": report.avg_latency_ms,
                }
            )
        self.logger.info(
            "Evaluation finished: avg_overall=%.2f, tokens=%d, cost=%.6f, avg_latency_ms=%.1f",
            report.avg_overall,
            report.total_tokens,
            report.total_cost_usd,
            report.avg_latency_ms,
        )
        return report

    def _print_summary(self, report: EvalReport):
        console.print(
            f"[dim]judge: {report.judge_model} · dataset: {report.dataset or '-'} "
            f"· {report.created_at}[/dim]"
        )
        table = Table(title=f"Results — {report.model}")
        table.add_column("Dimension", style="cyan")
        table.add_column("Average Score", style="green")

        for dim in ["faithfulness", "relevance", "clarity", "safety", "consistency"]:
            score = getattr(report, f"avg_{dim}")
            color = "green" if score >= 0.8 else "yellow" if score >= 0.6 else "red"
            table.add_row(dim, f"[{color}]{score:.2f}[/{color}]")

        table.add_row("─" * 15, "─" * 15)
        table.add_row("[bold]overall[/bold]", f"[bold]{report.avg_overall:.2f}[/bold]")
        table.add_row("tokens", str(report.total_tokens))
        table.add_row("cost_usd", f"${report.total_cost_usd:.6f}")
        table.add_row("latency_ms", f"{report.avg_latency_ms:.1f}")
        console.print(table)
