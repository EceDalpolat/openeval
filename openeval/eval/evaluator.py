# openeval/eval/evaluator.py

from rich.console import Console
from rich.table import Table

from ..connectors.base import BaseConnector
from ..judge.judge import Judge
from ..judge.schemas import EvalCase, EvalReport
from ..observability import SessionMetrics, get_logger, tracer

console = Console()

class Evaluator:
    """
    Ana sınıf. Kullanıcı sadece bunu kullanır.
    
    Örnek:
        evaluator = Evaluator(connector=OpenAIConnector())
        report = evaluator.run(cases)
    """

    def __init__(
        self,
        connector: BaseConnector,
        judge_connector: BaseConnector | None = None,
        tracer_client: object | None = None,
    ):
        self.connector = connector
        self.logger = get_logger(__name__)
        self.metrics = SessionMetrics()
        self.tracer = tracer_client or tracer
        # Judge için ayrı model kullanabilirsin (daha güçlü biri)
        # Verilmezse aynı connector kullanılır
        self.judge = Judge(
            judge_connector or connector,
            metrics=self.metrics,
            tracer_client=self.tracer,
        )

    def run(self, cases: list[EvalCase]) -> EvalReport:
        results = []

        self.logger.info("Evaluation başladı: cases=%d, model=%s", len(cases), self.connector.model_name)
        if getattr(self.tracer, "start_trace", None):
            self.tracer.start_trace(
                name="openeval.run",
                metadata={
                    "model": self.connector.model_name,
                    "judge_model": (self.judge.connector.model_name if self.judge.connector else self.connector.model_name),
                    "total_cases": len(cases),
                },
            )

        console.print(f"\n[bold]OpenEval[/bold] — {len(cases)} vaka değerlendiriliyor\n")

        for i, case in enumerate(cases, 1):
            console.print(f"[{i}/{len(cases)}] {case.question[:60]}...")
            self.logger.info("Case %d/%d işleniyor", i, len(cases))

            # 1. Modelden cevap al (connector verilmemişse case.answer kullanılır)
            result = self.judge.evaluate(case)
            results.append(result)
            console.print(f"  → overall: [green]{result.overall_score:.2f}[/green]")
            self.logger.info("Case %d tamamlandı: overall=%.2f", i, result.overall_score)

        # Ortalamalar
        avg = lambda dim: sum(getattr(r, dim).score for r in results) / len(results)

        report = EvalReport(
            model=self.connector.model_name,
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
            "Evaluation bitti: avg_overall=%.2f, tokens=%d, cost=%.6f, avg_latency_ms=%.1f",
            report.avg_overall,
            report.total_tokens,
            report.total_cost_usd,
            report.avg_latency_ms,
        )
        return report

    def _print_summary(self, report: EvalReport):
        table = Table(title=f"Sonuçlar — {report.model}")
        table.add_column("Boyut", style="cyan")
        table.add_column("Ortalama Skor", style="green")

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
