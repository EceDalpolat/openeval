# openeval/judge/schemas.py

from pydantic import BaseModel, Field

class DimensionScore(BaseModel):
    score: float = Field(ge=0.0, le=1.0, description="between 0.0 and 1.0")
    reasoning: str = Field(description="Why did you give this score?")

class EvaluationResult(BaseModel):
    """
    The judge's full evaluation for a single question-answer pair.
    A general version of the 5-dimension logic from the Referee Agent.
    """
    faithfulness: DimensionScore    # Does the answer match the facts?
    relevance: DimensionScore       # Is it relevant to the question?
    clarity: DimensionScore         # Is it clear?
    safety: DimensionScore          # Is there any harmful content?
    consistency: DimensionScore     # Is it internally consistent?

    @property
    def overall_score(self) -> float:
        """Weighted average — faithfulness and relevance matter more."""
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
    """A single test case."""
    question: str
    answer: str
    context: str | None = None      # retrieved text, if RAG is used

class EvalReport(BaseModel):
    """Summary of the whole evaluation."""
    model: str                      # the system that produced the answers (subject)
    judge_model: str = ""           # the model that did the scoring
    dataset: str | None = None      # which dataset was scored
    created_at: str = ""            # ISO timestamp — when the run happened
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
