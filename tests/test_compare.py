# Tests for compare.diff_reports.

from openeval.compare import diff_reports
from openeval.judge.schemas import EvalReport


def _report(**avgs) -> EvalReport:
    """Builds a minimal report with only the average scores (for testing)."""
    base = dict(
        faithfulness=0.5, relevance=0.5, clarity=0.5,
        safety=0.5, consistency=0.5, overall=0.5,
    )
    base.update(avgs)
    return EvalReport(
        model="test",
        total_cases=0,
        results=[],
        **{f"avg_{k}": v for k, v in base.items()},
    )


def test_diff_reports_computes_delta():
    before = _report(faithfulness=0.71, overall=0.70)
    after = _report(faithfulness=0.88, overall=0.85)
    rows = {r["dimension"]: r for r in diff_reports(before, after)}

    assert rows["faithfulness"]["before"] == 0.71
    assert rows["faithfulness"]["after"] == 0.88
    assert rows["faithfulness"]["delta"] == 0.17   # improvement
    assert rows["overall"]["delta"] == 0.15


def test_diff_reports_negative_delta():
    rows = {r["dimension"]: r for r in diff_reports(_report(safety=0.9), _report(safety=0.6))}
    assert rows["safety"]["delta"] == -0.3         # regression


def test_diff_reports_covers_all_dimensions():
    rows = diff_reports(_report(), _report())
    dims = {r["dimension"] for r in rows}
    assert dims == {"faithfulness", "relevance", "clarity", "safety", "consistency", "overall"}
