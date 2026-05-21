# openeval/connectors/ollama_connector.py

import httpx
from .base import BaseConnector, ModelResponse

class OllamaConnector(BaseConnector):
    """
    Ollama connector — lokal modeller için (Llama3, Mistral, vb.)
    
    Kurulum:
        brew install ollama           # macOS
        ollama pull llama3.2          # modeli indir
        ollama serve                  # servisi başlat
    
    Ollama, localhost:11434'te REST API sunar.
    OpenAI ile aynı interface — sadece URL değişiyor.
    """

    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434"):
        self._model = model
        self._base_url = base_url

    @property
    def model_name(self) -> str:
        return f"ollama/{self._model}"

    def is_available(self) -> bool:
        try:
            r = httpx.get(f"{self._base_url}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def generate(self, prompt: str, system: str = "") -> ModelResponse:
        payload = {
            "model": self._model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"temperature": 0},
        }

        r = httpx.post(
            f"{self._base_url}/api/generate",
            json=payload,
            timeout=120,    # lokal model yavaş olabilir
        )
        r.raise_for_status()
        data = r.json()

        return ModelResponse(
            content=data["response"],
            model=self._model,
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
        )