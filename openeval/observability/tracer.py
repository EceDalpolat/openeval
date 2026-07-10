# openeval/observability/tracer.py

import os
from contextlib import contextmanager
from .logger import get_logger
from .metrics import CallMetrics

logger = get_logger(__name__)

# Langfuse is optional — if it is not installed, tracing is silently disabled
try:
    from langfuse import Langfuse
    _langfuse_available = True
except ImportError:
    _langfuse_available = False
    logger.debug("Langfuse not found. Tracing disabled.")


class Tracer:
    """
    Langfuse tracing wrapper for OpenEval.

    Tracing is active if Langfuse is installed and the env vars are set.
    If it is not installed, nothing crashes — a log message is written instead.

    Environment variables (.env):
        LANGFUSE_PUBLIC_KEY=pk-lf-...
        LANGFUSE_SECRET_KEY=sk-lf-...
        LANGFUSE_HOST=https://cloud.langfuse.com   # or self-hosted
    """

    def __init__(self):
        self._client = None
        self._active_trace = None

        if _langfuse_available and self._has_credentials():
            try:
                self._client = Langfuse(
                    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
                    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
                    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
                )
                logger.info("Langfuse tracing active ✓")
            except Exception as e:
                logger.warning(f"Could not initialize Langfuse: {e}")

    def _has_credentials(self) -> bool:
        return bool(
            os.getenv("LANGFUSE_PUBLIC_KEY") and
            os.getenv("LANGFUSE_SECRET_KEY")
        )

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def start_trace(self, name: str, metadata: dict | None = None):
        """Start a trace for a new eval session."""
        if not self.enabled:
            return

        self._active_trace = self._client.trace(
            name=name,
            metadata=metadata or {},
            tags=["openeval"],
        )
        logger.debug(f"Trace started: {name}")

    def log_generation(
        self,
        name: str,
        prompt: str,
        response: str,
        metrics: CallMetrics,
        metadata: dict | None = None,
    ):
        """Log a single LLM call to Langfuse."""
        if not self.enabled or not self._active_trace:
            return

        try:
            self._active_trace.generation(
                name=name,
                model=metrics.model,
                input=prompt,
                output=response,
                usage={
                    "input":  metrics.input_tokens,
                    "output": metrics.output_tokens,
                    "total":  metrics.total_tokens,
                },
                metadata={
                    "latency_ms": metrics.latency_ms,
                    "cost_usd":   metrics.cost_usd,
                    **(metadata or {}),
                },
            )
        except Exception as e:
            logger.debug(f"Could not log Langfuse generation: {e}")

    def log_score(self, name: str, value: float, comment: str = ""):
        """Add a dimension score to the trace."""
        if not self.enabled or not self._active_trace:
            return
        try:
            self._active_trace.score(name=name, value=value, comment=comment)
        except Exception as e:
            logger.debug(f"Could not log score: {e}")

    def end_trace(self, metadata: dict | None = None):
        """Close the trace and flush it to Langfuse."""
        if not self.enabled or not self._active_trace:
            return
        try:
            if metadata:
                self._active_trace.update(metadata=metadata)
            self._client.flush()
            logger.debug("Trace closed and flushed")
        except Exception as e:
            logger.debug(f"Could not close trace: {e}")


# Singleton — the whole project uses the same tracer
tracer = Tracer()
