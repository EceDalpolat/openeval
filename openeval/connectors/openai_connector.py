# openeval/connectors/openai_connector.py

import os
from openai import OpenAI
from .base import BaseConnector, ModelResponse

class OpenAIConnector(BaseConnector):
    """
    OpenAI API connector.
    GPT-4o, GPT-4o-mini ve diğer OpenAI modelleri için.
    """

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        self._model = model
        self._client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

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

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0,      # eval için deterministik olsun
        )

        return ModelResponse(
            content=response.choices[0].message.content,
            model=self._model,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )