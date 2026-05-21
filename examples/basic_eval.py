# examples/basic_eval.py — şu anki dosyanı şununla değiştir

from dotenv import load_dotenv
load_dotenv()

from openeval.connectors.openrouter_connector import OpenRouterConnector
from openeval.connectors.ollama_connector import OllamaConnector
from openeval.eval.evaluator import Evaluator
from openeval.judge.schemas import EvalCase

cases = [
    EvalCase(
        question="Python'da liste nasıl sıralanır?",
        answer="sorted() veya .sort() kullanılır.",
    ),
    EvalCase(
        question="FastAPI nedir?",
        answer="Python ile API geliştirmek için kullanılan framework.",
    ),
]

# OpenRouter ile test et, Ollama ile judge yap
evaluator = Evaluator(
    connector=OpenRouterConnector(),   # ücretsiz model
    judge_connector=OllamaConnector(model="llama3.2:3b"),  # lokal
)

report = evaluator.run(cases)