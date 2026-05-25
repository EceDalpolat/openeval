from dotenv import load_dotenv
load_dotenv()

from openeval.connectors.openrouter_connector import OpenRouterConnector
from openeval.connectors.ollama_connector import OllamaConnector
from openeval.eval.evaluator import Evaluator
from openeval.judge.schemas import EvalCase
from openeval.rag import TFIDFRetriever
import json, pathlib

# RAG retriever — knowledge base'i yukle
retriever = TFIDFRetriever(top_k=2)

def make_case(question: str, answer: str) -> EvalCase:
    """Soruya gore otomatik context ekle."""
    context = retriever.retrieve_as_context(question)
    return EvalCase(question=question, answer=answer, context=context)

cases = [
    # Dogru cevaplar
    make_case(
        "RAG nedir ve ne zaman kullanılır?",
        "RAG, LLM'e harici bilgi kaynağı ekler. Modelin bilmediği "
        "güncel bilgileri sorgu anında vermek için kullanılır.",
    ),
    make_case(
        "Embedding nedir?",
        "Metni sayısal vektöre dönüştürme işlemidir. "
        "Benzer anlamlı metinler uzayda birbirine yakın durur.",
    ),
    make_case(
        "Fine-tuning ile RAG arasındaki fark nedir?",
        "RAG güncel/dinamik bilgiler için, fine-tuning ise "
        "modelin davranışını veya tonunu değiştirmek için kullanılır.",
    ),
    make_case(
        "LLM'de hallucination nedir?",
        "Modelin yanlış ama emin görünen bilgi üretmesidir.",
    ),
    make_case(
        "LoRA nedir?",
        "Modelin tüm parametrelerini değil, küçük adapter "
        "katmanlarını eğiten fine-tuning yöntemidir.",
    ),
    # Eksik/yanlis cevaplar — judge yakalamali
    make_case(
        "Vektör veritabanı nedir?",
        "Bir tür veritabanıdır.",   # cok yuzeysel
    ),
    make_case(
        "Context window nedir?",
        "Modelin hafızasıdır.",     # eksik tanim
    ),
    make_case(
        "Temperature parametresi ne işe yarar?",
        "Temperature yüksekse model daha hızlı çalışır.",  # yanlis!
    ),
]

print(f"\n📚 RAG aktif — {len(cases)} vaka, knowledge base yuklu\n")

evaluator = Evaluator(
    connector=OpenRouterConnector(
        model="meta-llama/llama-3.2-3b-instruct:free"
    ),
    judge_connector=OllamaConnector(model="llama3.2:3b"),
)

report = evaluator.run(cases)

# Raporu kaydet
pathlib.Path("reports").mkdir(exist_ok=True)
with open("reports/eval_report.json", "w", encoding="utf-8") as f:
    f.write(report.model_dump_json(indent=2))

print("\n✅ Rapor kaydedildi: reports/eval_report.json")
