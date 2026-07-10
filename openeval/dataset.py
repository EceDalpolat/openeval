# openeval/dataset.py
#
# JSONL dataset loader — bir .jsonl dosyasını okuyup EvalCase listesine çevirir.
#
# JSONL = "JSON Lines": her SATIR ayrı bir JSON objesi. Örnek bir satır:
#   {"question": "RAG nedir?", "answer": "...", "context": "..."}
#
# Neden JSONL? Çünkü herhangi bir proje (agentic-rag, finance) çıktısını satır satır
# bu dosyaya döküp openeval'e "şunu puanla" diyebilir. Elle Python yazmaya son.

import json
from pathlib import Path

from .judge.schemas import EvalCase


def load_cases(path: str | Path) -> list[EvalCase]:
    """
    Bir .jsonl dosyasını okuyup EvalCase listesi döndürür.

    Her satır şu alanları içermeli:
      - question (zorunlu): sorulan soru
      - answer   (zorunlu): sistemin verdiği cevap (openeval bunu puanlar)
      - context  (opsiyonel): RAG varsa retrieval edilen metin

    Hatalı satırda, hangi satırda ne olduğunu net söyler (sessizce atlamaz).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset bulunamadı: {path}")

    cases: list[EvalCase] = []

    with path.open(encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw or raw.startswith("#"):  # boş satır / yorum → atla
                continue

            try:
                data = json.loads(raw)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"{path}:{line_no} geçerli JSON değil → {e.msg}"
                ) from e

            if "question" not in data or "answer" not in data:
                raise ValueError(
                    f"{path}:{line_no} 'question' ve 'answer' alanları zorunlu. "
                    f"Gelen alanlar: {list(data.keys())}"
                )

            cases.append(
                EvalCase(
                    question=data["question"],
                    answer=data["answer"],
                    context=data.get("context"),
                )
            )

    if not cases:
        raise ValueError(f"{path} içinde geçerli vaka yok (dosya boş mu?)")

    return cases
