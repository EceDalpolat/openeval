# openeval/connectors/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class ModelResponse:
    """The response from every connector comes in this format."""
    content: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0

class BaseConnector(ABC):
    """
    The interface every model connector must implement.
    You cannot use this class directly — it is only meant to be subclassed.
    """

    @abstractmethod
    def generate(self, prompt: str, system: str = "") -> ModelResponse:
        """Send a prompt to the model and get a response."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Is the model reachable? (Is there an API key, is the service running?)"""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The model's name."""
        ...