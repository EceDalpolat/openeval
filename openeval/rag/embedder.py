"""
Ollama nomic-embed-text ile lokal embedding.
Tamamen ucretsiz, M4'te hizli calisir.
"""
import httpx
from ..observability import get_logger

logger = get_logger(__name__)


class OllamaEmbedder:
    """
    Metni 768 boyutlu vektore donusturur.
    Ollama'nin nomic-embed-text modelini kullanir.
    
    Nasil calisir:
      "RAG nedir?" → [0.82, -0.14, 0.56, ...] (768 sayi)
      "Retrieval Augmented Generation" → [0.81, -0.13, 0.57, ...]
      Bu iki vektor birbirine cok yakin → anlam benzer!
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
    ):
        self.model = model
        self.base_url = base_url

    def embed(self, text: str) -> list[float]:
        """Tek bir metni vektore cevir."""
        response = httpx.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=30,
        )
        response.raise_for_status()
        vector = response.json()["embedding"]
        logger.debug("Embedded: %d chars → %d dim vector", len(text), len(vector))
        return vector

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Birden fazla metni vektore cevir."""
        logger.info("Embedding %d metin...", len(texts))
        return [self.embed(t) for t in texts]

    def is_available(self) -> bool:
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False
