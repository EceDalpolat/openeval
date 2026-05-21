# openeval/connectors/openrouter_connector.py
import os
from openai import OpenAI
from .base import BaseConnector, ModelResponse

class OpenRouterConnector(BaseConnector):
    def __init__(self, model: str = "meta-llama/llama-3.2-3b-instruct:free"):
        self._model = model
        self._client = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )

    @property
    def model_name(self): return self._model

    def is_available(self): return bool(os.getenv("OPENROUTER_API_KEY"))

    def generate(self, prompt: str, system: str = "") -> ModelResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0,
        )
        return ModelResponse(
            content=response.choices[0].message.content,
            model=self._model,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )