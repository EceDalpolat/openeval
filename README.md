# OpenEval

OpenEval, LLM çıktılarının otomatik olarak değerlendirilmesi için hafif bir Python çerçevesidir.  
OpenAI API veya yerel Ollama modelleriyle çalışır ve bir değerlendirme raporu üretir.

## Özellikler

- `OpenAIConnector` ile OpenAI modellerine bağlanır
- `OllamaConnector` ile yerel modelleri kullanır
- `Evaluator` ile soru/cevap vakalarını toplu değerlendirir
- `LLM-as-judge` yaklaşımıyla 5 boyutta skor üretir:
  - `faithfulness`
  - `relevance`
  - `clarity`
  - `safety`
  - `consistency`

## Kurulum

Python 3.11 veya üzeri gerekir.

```bash
pip install -e .
```

Geliştirme bağımlılıkları için:

```bash
pip install -e ".[dev]"
```

## Ortam değişkenleri

OpenAI kullanıyorsan `OPENAI_API_KEY` tanımla:

```bash
cp .env.example .env
```

Ardından `.env` içine anahtarını ekle.

## Hızlı başlangıç

`examples/basic_eval.py` dosyasını örnek olarak kullanabilirsin:

```python
from openeval.connectors.openai_connector import OpenAIConnector
from openeval.eval.evaluator import Evaluator
from openeval.judge.schemas import EvalCase

cases = [
    EvalCase(
        question="Python'da bir listeyi tersine nasıl çevirirsin?",
        answer="liste.reverse() metodunu veya liste[::-1] slice'ını kullanabilirsin.",
    ),
]

connector = OpenAIConnector(model="gpt-4o-mini")
evaluator = Evaluator(connector=connector)
report = evaluator.run(cases)
```

## Ollama ile kullanım

Yerel model çalıştırmak için:

```bash
ollama serve
ollama pull llama3.2
```

Sonra:

```python
from openeval.connectors.ollama_connector import OllamaConnector
from openeval.eval.evaluator import Evaluator

connector = OllamaConnector(model="llama3.2")
evaluator = Evaluator(connector=connector)
```

## Çıktı

`Evaluator.run(...)` bir `EvalReport` döndürür. Bu rapor:

- model adını
- toplam vaka sayısını
- her vaka için detaylı `EvaluationResult` listesini
- boyut bazlı ortalama skorları

içerir.

## Proje yapısı

```text
openeval/
├── connectors/   # model sağlayıcıları
├── eval/         # ana değerlendirme akışı
├── judge/        # skorlayan LLM judge mantığı
├── report/       # rapor yardımcıları
└── examples/     # kullanım örnekleri
```

## Notlar

- Judge modeli, varsayılan olarak değerlendirme connector’ı ile aynı olabilir.
- Daha iyi sonuç için değerlendirme yapan model ile judge modelini ayırabilirsin.
- Model çıktısı JSON beklenir; judge cevabı bunu bozarsa parse hatası alınır.

