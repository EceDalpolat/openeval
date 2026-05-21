# openeval/connectors/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class ModelResponse:
    """Her connector'dan gelen yanıt bu formatta olur."""
    content: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0

class BaseConnector(ABC):
    """
    Tüm model connector'larının implement etmesi gereken interface.
    Bu sınıfı direkt kullanamazsın — sadece miras alınır.
    """

    @abstractmethod
    def generate(self, prompt: str, system: str = "") -> ModelResponse:
        """Modele prompt gönder, cevap al."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Model erişilebilir mi? (API key var mı, servis çalışıyor mu?)"""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Modelin adı."""
        ...