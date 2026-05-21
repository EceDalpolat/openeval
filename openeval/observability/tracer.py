# openeval/observability/tracer.py

import os
from contextlib import contextmanager
from .logger import get_logger
from .metrics import CallMetrics

logger = get_logger(__name__)

# Langfuse opsiyonel — kurulmamışsa tracing sessizce devre dışı kalır
try:
    from langfuse import Langfuse
    _langfuse_available = True
except ImportError:
    _langfuse_available = False
    logger.debug("Langfuse bulunamadı. Tracing devre dışı.")


class Tracer:
    """
    OpenEval için Langfuse tracing wrapper.
    
    Langfuse kurulu ve env var'lar tanımlıysa tracing aktif.
    Kurulu değilse hiçbir şey patlamaz, sadece log yazılır.
    
    Ortam değişkenleri (.env):
        LANGFUSE_PUBLIC_KEY=pk-lf-...
        LANGFUSE_SECRET_KEY=sk-lf-...
        LANGFUSE_HOST=https://cloud.langfuse.com   # veya self-hosted
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
                logger.info("Langfuse tracing aktif ✓")
            except Exception as e:
                logger.warning(f"Langfuse başlatılamadı: {e}")

    def _has_credentials(self) -> bool:
        return bool(
            os.getenv("LANGFUSE_PUBLIC_KEY") and
            os.getenv("LANGFUSE_SECRET_KEY")
        )

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def start_trace(self, name: str, metadata: dict | None = None):
        """Yeni bir eval oturumu için trace başlat."""
        if not self.enabled:
            return

        self._active_trace = self._client.trace(
            name=name,
            metadata=metadata or {},
            tags=["openeval"],
        )
        logger.debug(f"Trace başladı: {name}")

    def log_generation(
        self,
        name: str,
        prompt: str,
        response: str,
        metrics: CallMetrics,
        metadata: dict | None = None,
    ):
        """Tek bir LLM çağrısını Langfuse'a kaydet."""
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
            logger.debug(f"Langfuse generation kaydedilemedi: {e}")

    def log_score(self, name: str, value: float, comment: str = ""):
        """Boyut skorunu trace'e ekle."""
        if not self.enabled or not self._active_trace:
            return
        try:
            self._active_trace.score(name=name, value=value, comment=comment)
        except Exception as e:
            logger.debug(f"Score kaydedilemedi: {e}")

    def end_trace(self, metadata: dict | None = None):
        """Trace'i kapat ve Langfuse'a flush et."""
        if not self.enabled or not self._active_trace:
            return
        try:
            if metadata:
                self._active_trace.update(metadata=metadata)
            self._client.flush()
            logger.debug("Trace kapatıldı ve flush edildi")
        except Exception as e:
            logger.debug(f"Trace kapatılamadı: {e}")


# Singleton — tüm proje aynı tracer'ı kullanır
tracer = Tracer()
