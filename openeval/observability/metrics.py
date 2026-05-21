# openeval/observability/metrics.py

import time
from dataclasses import dataclass, field
from typing import ClassVar

# OpenAI fiyatları (Mayıs 2026, gpt-4o-mini)
COST_PER_1K_TOKENS: dict[str, dict] = {
    "gpt-4o-mini":  {"input": 0.000150, "output": 0.000600},
    "gpt-4o":       {"input": 0.002500, "output": 0.010000},
    "default":      {"input": 0.000150, "output": 0.000600},
}

@dataclass
class CallMetrics:
    """Tek bir LLM çağrısının metrikleri."""
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost_usd(self) -> float:
        pricing = COST_PER_1K_TOKENS.get(self.model, COST_PER_1K_TOKENS["default"])
        return (
            self.input_tokens  / 1000 * pricing["input"] +
            self.output_tokens / 1000 * pricing["output"]
        )

    def __str__(self) -> str:
        return (
            f"model={self.model} | "
            f"tokens={self.total_tokens} ({self.input_tokens}in/{self.output_tokens}out) | "
            f"latency={self.latency_ms:.0f}ms | "
            f"cost=${self.cost_usd:.6f}"
        )


@dataclass
class SessionMetrics:
    """
    Tüm eval oturumunun birikimli metrikleri.
    Evaluator.run() boyunca birikiyor.
    """
    calls: list[CallMetrics] = field(default_factory=list)

    def add(self, m: CallMetrics) -> None:
        self.calls.append(m)

    @property
    def total_tokens(self) -> int:
        return sum(c.total_tokens for c in self.calls)

    @property
    def total_cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.calls)

    @property
    def avg_latency_ms(self) -> float:
        if not self.calls:
            return 0.0
        return sum(c.latency_ms for c in self.calls) / len(self.calls)

    def summary(self) -> dict:
        return {
            "total_calls":    len(self.calls),
            "total_tokens":   self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
        }


class Timer:
    """Context manager ile latency ölç."""

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000

# Kullanım:
# with Timer() as t:
#     response = connector.generate(prompt)
# print(t.elapsed_ms)  # → 1243.7