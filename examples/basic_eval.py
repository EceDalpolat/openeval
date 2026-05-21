# examples/basic_eval.py

import os
from dotenv import load_dotenv
load_dotenv()

from openeval.connectors.openai_connector import OpenAIConnector
from openeval.eval.evaluator import Evaluator
from openeval.judge.schemas import EvalCase
import json

# Test vakalarını tanımla
cases = [
    EvalCase(
        question="Python'da bir listeyi tersine nasıl çevirirsin?",
        answer="liste.reverse() metodunu veya liste[::-1] slice'ını kullanabilirsin.",
    ),
    EvalCase(
        question="FastAPI nedir?",
        answer="FastAPI, Python ile hızlı API geliştirmek için kullanılan modern bir framework'tür.",
    ),
    EvalCase(
        question="İstanbul'un nüfusu nedir?",
        answer="İstanbul'un nüfusu yaklaşık 15 milyondur.",
    ),
]

# Değerlendirmeyi çalıştır
connector = OpenAIConnector(model="gpt-4o-mini")
evaluator = Evaluator(connector=connector)
report = evaluator.run(cases)

# JSON olarak kaydet
with open("report.json", "w", encoding="utf-8") as f:
    f.write(report.model_dump_json(indent=2))

print("\nrapor report.json'a kaydedildi.")