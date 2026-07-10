from dotenv import load_dotenv
load_dotenv()

from openeval.connectors.openrouter_connector import OpenRouterConnector
from openeval.connectors.ollama_connector import OllamaConnector
from openeval.eval.evaluator import Evaluator
from openeval.judge.schemas import EvalCase
from openeval.rag import ChromaRetriever
import json, pathlib

# RAG retriever — load the knowledge base
retriever = ChromaRetriever(top_k=2)

def make_case(question: str, answer: str) -> EvalCase:
    """Automatically add context based on the question."""
    context = retriever.retrieve_as_context(question)
    return EvalCase(question=question, answer=answer, context=context)

cases = [
    # Correct answers
    make_case(
        "What is RAG and when is it used?",
        "RAG adds an external knowledge source to an LLM. It is used to provide "
        "up-to-date information the model does not know at query time.",
    ),
    make_case(
        "What is embedding?",
        "It is the process of converting text into a numeric vector. "
        "Semantically similar texts end up close to each other in the space.",
    ),
    make_case(
        "What is the difference between fine-tuning and RAG?",
        "RAG is for up-to-date/dynamic information, while fine-tuning is "
        "for changing the model's behavior or tone.",
    ),
    make_case(
        "What is hallucination in an LLM?",
        "It is the model producing wrong but confident-looking information.",
    ),
    make_case(
        "What is LoRA?",
        "It is a fine-tuning method that trains small adapter layers "
        "instead of all of the model's parameters.",
    ),
    # Incomplete/wrong answers — the judge should catch these
    make_case(
        "What is a vector database?",
        "It is a kind of database.",   # very shallow
    ),
    make_case(
        "What is a context window?",
        "It is the model's memory.",     # incomplete definition
    ),
    make_case(
        "What does the temperature parameter do?",
        "If temperature is high, the model runs faster.",  # wrong!
    ),
]

print(f"\n📚 RAG active — {len(cases)} cases, knowledge base loaded\n")

evaluator = Evaluator(
    connector=OpenRouterConnector(
        model="meta-llama/llama-3.2-3b-instruct:free"
    ),
    judge_connector=OllamaConnector(model="llama3.2:3b"),
)

report = evaluator.run(cases)

# Save the report
pathlib.Path("reports").mkdir(exist_ok=True)
with open("reports/eval_report.json", "w", encoding="utf-8") as f:
    f.write(report.model_dump_json(indent=2))

print("\n✅ Report saved: reports/eval_report.json")
