# openeval/eval/evaluator.py

from rich.console import Console
from rich.table import Table
from ..connectors.base import BaseConnector
from ..judge.judge import Judge
from ..judge.schemas import EvalCase, EvalReport

console = Console()

class Evaluator:
    """
    Ana sınıf. Kullanıcı sadece bunu kullanır.
    
    Örnek:
        evaluator = Evaluator(connector=OpenAIConnector())
        report = evaluator.run(cases)
    """

    def __init__(self, connector: BaseConnector, judge_connector: BaseConnector | None = None):
        self.connector = connector
        # Judge için ayrı model kullanabilirsin (daha güçlü biri)
        # Verilmezse aynı connector kullanılır
        self.judge = Judge(judge_connector or connector)

    def run(self, cases: list[EvalCase]) -> EvalReport:
        results = []

        console.print(f"\n[bold]OpenEval[/bold] — {len(cases)} vaka değerlendiriliyor\n")

        for i, case in enumerate(cases, 1):
            console.print(f"[{i}/{len(cases)}] {case.question[:60]}...")

            # 1. Modelden cevap al (connector verilmemişse case.answer kullanılır)
            result = self.judge.evaluate(case)
            results.append(result)
            console.print(f"  → overall: [green]{result.overall_score:.2f}[/green]")

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
        )

        self._print_summary(report)
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
        console.print(table)