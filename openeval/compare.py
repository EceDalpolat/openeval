# openeval/compare.py
#
# İki raporu kıyaslar: before → after. Asıl değeri buradan alıyoruz —
# "faithfulness 0.71 → 0.88 (+0.17)" gibi before/after impact metrikleri.
# Örn: agentic-rag'de Auditor döngüsü KAPALI vs AÇIK.

from .judge.schemas import EvalReport

# overall dahil tüm boyutlar (hepsi rapor içinde avg_<dim> olarak duruyor)
DIMENSIONS = ["faithfulness", "relevance", "clarity", "safety", "consistency", "overall"]


def diff_reports(before: EvalReport, after: EvalReport) -> list[dict]:
    """
    Her boyut için before, after ve delta (after - before) döndürür.
    delta > 0  → iyileşme,  delta < 0 → kötüleşme.
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
