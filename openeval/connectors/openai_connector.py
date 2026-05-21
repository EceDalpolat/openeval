# openeval/connectors/openai_connector.py

import os

from openai import OpenAI

from ..observability import CallMetrics, Timer, get_logger, tracer
from .base import BaseConnector, ModelResponse

class OpenAIConnector(BaseConnector):
    """
    OpenAI API connector.
    GPT-4o, GPT-4o-mini ve diğer OpenAI modelleri için.
    """

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        self._model = model
        self._client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.logger = get_logger(__name__)
        self.tracer = tracer

    @property
    def model_name(self) -> str:
        return self._model

    def is_available(self) -> bool:
        return bool(os.getenv("OPENAI_API_KEY"))

    def generate(self, prompt: str, system: str = "") -> ModelResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        self.logger.info("OpenAI generate başladı: model=%s", self._model)

        with Timer() as timer:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=0,      # eval için deterministik olsun
            )

        input_tokens = getattr(response.usage, "prompt_tokens", 0) if response.usage else 0
        output_tokens = getattr(response.usage, "completion_tokens", 0) if response.usage else 0
        content = response.choices[0].message.content or ""
        call_metrics = CallMetrics(
            model=self._model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=timer.elapsed_ms,
        )

        if getattr(self.tracer, "log_generation", None):
            self.tracer.log_generation(
                name="openai.generate",
                prompt=prompt,
                response=content,
                metrics=call_metrics,
                metadata={
                    "system_prompt": bool(system),
                    "message_count": len(messages),
                },
            )

        self.logger.info(
            "OpenAI generate tamamlandı: model=%s, tokens=%d, latency_ms=%.1f",
            self._model,
            call_metrics.total_tokens,
            call_metrics.latency_ms,
        )

        return ModelResponse(
            content=content,
            model=self._model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
