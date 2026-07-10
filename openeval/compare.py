# openeval/compare.py
#
# Compares two reports: before → after. This is where the real value comes from —
# before/after impact metrics like "faithfulness 0.71 → 0.88 (+0.17)".
# For example: Auditor loop OFF vs ON in agentic-rag.

from .judge.schemas import EvalReport

# all dimensions including overall (each stored as avg_<dim> in the report)
DIMENSIONS = ["faithfulness", "relevance", "clarity", "safety", "consistency", "overall"]


def diff_reports(before: EvalReport, after: EvalReport) -> list[dict]:
    """
    Returns before, after and delta (after - before) for each dimension.
    delta > 0  → improvement,  delta < 0 → regression.
    """
    rows = []
    for dim in DIMENSIONS:
        b = getattr(before, f"avg_{dim}")
        a = getattr(after, f"avg_{dim}")
        rows.append(
            {
                "dimension": dim,
                "before": round(b, 4),
                "after": round(a, 4),
                "delta": round(a - b, 4),
            }
        )
    return rows
