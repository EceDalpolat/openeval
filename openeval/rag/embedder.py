"""
Local embedding with Ollama's nomic-embed-text.
Completely free, runs fast on the M4.
"""
import httpx
from ..observability import get_logger

logger = get_logger(__name__)


class OllamaEmbedder:
    """
    Converts text into a 768-dimensional vector.
    Uses Ollama's nomic-embed-text model.

    How it works:
      "What is RAG?" → [0.82, -0.14, 0.56, ...] (768 numbers)
      "Retrieval Augmented Generation" → [0.81, -0.13, 0.57, ...]
      These two vectors are very close → the meanings are similar!
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
    ):
        self.model = model
        self.base_url = base_url

    def embed(self, text: str) -> list[float]:
        """Convert a single piece of text into a vector."""
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
        """Convert multiple pieces of text into vectors."""
        logger.info("Embedding %d texts...", len(texts))
        return [self.embed(t) for t in texts]

    def is_available(self) -> bool:
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False
