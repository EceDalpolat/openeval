# openeval/judge/schemas.py

from pydantic import BaseModel, Field

class DimensionScore(BaseModel):
    score: float = Field(ge=0.0, le=1.0, description="0.0 ile 1.0 arası")
    reasoning: str = Field(description="Neden bu skoru verdin?")

class EvaluationResult(BaseModel):
    """
    Her soru-cevap çifti için judge'ın verdiği tam değerlendirme.
    Referee Agent'ındaki 5-boyut mantığının genel versiyonu.
    """
    faithfulness: DimensionScore    # Cevap gerçekle uyuşuyor mu?
    relevance: DimensionScore       # Soruyla ilgili mi?
    clarity: DimensionScore         # Anlaşılır mı?
    safety: DimensionScore          # Zararlı içerik var mı?
    consistency: DimensionScore     # Kendi içinde tutarlı mı?

    @property
    def overall_score(self) -> float:
        """Ağırlıklı ortalama — faithfulness ve relevance daha önemli."""
        weights = {
            "faithfulness": 0.30,
            "relevance":    0.30,
            "clarity":      0.20,
            "safety":       0.10,
            "consistency":  0.10,
        }
        return sum(
            getattr(self, dim).score * w
            for dim, w in weights.items()
        )

class EvalCase(BaseModel):
    """Tek bir test vakası."""
    question: str
    answer: str
    context: str | None = None      # RAG varsa retrieval edilen metin

class EvalReport(BaseModel):
    """Tüm değerlendirmenin özeti."""
    model: str
    total_cases: int
    results: list[EvaluationResult]
    avg_overall: float
    avg_faithfulness: float
    avg_relevance: float
    avg_clarity: float
    avg_safety: float
    avg_consistency: float
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    avg_latency_ms: float = 0.0
