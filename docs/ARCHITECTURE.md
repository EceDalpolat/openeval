# openeval — Mimari

openeval, bir LLM'in cevaplarını başka bir LLM'e ("hakem") not verdirerek ölçen
hafif bir değerlendirme aracıdır. Yöntem literatürde **LLM-as-judge** olarak geçer.

**Neden?** "Sistemim iyi cevap veriyor" demek yetmez — ölçmek gerekir. Binlerce cevabı
elle okumak yerine, güçlü bir modeli hakem yapıp her cevaba 5 boyutta 0–1 arası not
verdiririz.

---

## Veri akışı

```
dataset.jsonl                          [1] dataset.py: load_cases()
  │  {question, answer, context}
  ▼
list[EvalCase]                         [2] cli.py: judge connector'ı kur (ollama/openai/…)
  │
  ▼
Evaluator.run(cases)                   [3] eval/evaluator.py: orkestratör
  │  her vaka için ────────────┐
  ▼                            │
Judge.evaluate(case)           │       [4] judge/judge.py: asıl not veren
  │  prompt kur → judge modele │
  │  gönder → JSON'u parse et  │
  ▼                            │
EvaluationResult (5 boyut)  ◄──┘       [5] judge/schemas.py: skorlar + overall
  │
  ▼  (biriktir + ortalama al)
EvalReport  ──► reports/report.json    [6] künye + ortalamalar + maliyet
  │
  ▼
compare a.json b.json                  [7] compare.py: before/after Δ
```

---

## Bileşenler

### Şemalar — `judge/schemas.py`
Sistemin veri tipleri:

- **`EvalCase`** — tek test: `question`, `answer`, `context` (opsiyonel RAG metni).
- **`DimensionScore`** — tek boyutun notu: `score` (0.0–1.0) + `reasoning`.
- **`EvaluationResult`** — bir cevabın 5 boyutlu notu. `overall_score` **ağırlıklı**
  ortalamadır (aşağıdaki tablo). Cevap akıcı ama yanlışsa overall yine düşer.
- **`EvalReport`** — koşunun özeti: boyut ortalamaları + token/maliyet/latency +
  künye (`judge_model`, `dataset`, `created_at`).

### 5 boyut
| Boyut | Ölçtüğü | Ağırlık |
|---|---|---|
| faithfulness | Cevap doğru mu? | 0.30 |
| relevance | Soruyu cevaplıyor mu? | 0.30 |
| clarity | Anlaşılır mı? | 0.20 |
| safety | Zararlı/etik dışı mı? | 0.10 |
| consistency | Kendi içinde çelişiyor mu? | 0.10 |

### Connector'lar — `connectors/`
Modellerle konuşan katman. `BaseConnector` bir sözleşmedir (abstract): her connector
`generate(prompt)`, `is_available()`, `model_name` sağlar. İmplementasyonlar: OpenAI,
OpenRouter, Ollama. Aynı arayüz sayesinde judge modelini değiştirmek tek satırdır.

**subject vs judge — kritik ayrım:**
- **subject**: cevapları *üreten* sistem (ör. agentic-rag). openeval bunu **çalıştırmaz**,
  yalnızca adını etiket (`subject_label`) olarak tutar.
- **judge**: cevaplara *not veren* model — genelde subject'ten daha güçlü seçilir.

openeval hazır (pre-generated) cevapları puanladığı için subject connector zorunlu
değildir; yalnızca judge yeterlidir.

### Hakem — `judge/judge.py`
`evaluate(case)` adımları:
1. Prompt kurar (soru + cevap + context), "SADECE şu JSON'u döndür" talimatıyla.
2. Judge modele `temperature=0` ile gönderir → tekrarlanabilirlik.
3. Cevabı dayanıklı şekilde ayrıştırır:
   - `extract_json()` — model ```` ```fence ```` veya önsöz eklese bile JSON'u çeker.
   - `coerce_dimension()` — eksik/bozuk boyut → nötr 0.0 + not; `90→0.9`; 0–1'e sıkıştır.
   - retry — geçici API hatasında `MAX_RETRIES` kez dener.
4. `EvaluationResult` döndürür. Tek bir bozuk cevap tüm koşuyu çökertemez.

### Orkestratör — `eval/evaluator.py`
Vakaları tek tek hakeme verir, sonuçları toplar, ortalamaları hesaplar, künyeyi ve
metrikleri ekler, `EvalReport` üretir, özet tabloyu çizer. Kullanıcının dokunduğu ana
sınıf.

### Metrikler — `observability/metrics.py`
- `Timer` — çağrı süresi.
- `CallMetrics` — tek çağrının token/maliyet/latency'si. `cost_usd`, model adına göre
  fiyatlanır; `ollama/…` ve `local/…` modelleri $0 (lokal, ücretsiz).
- `SessionMetrics` — tüm koşunun toplamı.

### Loader / Compare / CLI
- `dataset.py` — JSONL → `list[EvalCase]`; bozuk satırda net hata verir.
- `compare.py` — `diff_reports(before, after)` boyut-boyut farkı hesaplar.
- `cli.py` — `run` (bir dataset'i puanla) ve `compare` (iki raporu kıyasla) komutları.

---

## Örnek: tek cevabın hikâyesi
Girdi: `"RAG nedir?"` + `"Bir şeydir işte."` →
Judge prompt'a gömülür → modele gider → judge döner:
`faithfulness 0.2, relevance 0.3, clarity 0.3, safety 1.0, consistency 0.6` →
overall = 0.2·0.3 + 0.3·0.3 + 0.3·0.2 + 1.0·0.1 + 0.6·0.1 ≈ **0.31** (düşük — boş cevap).

---

## Kullanım

```bash
# Bir dataset'i puanla (lokal Ollama hakemiyle)
openeval run examples/sample_cases.jsonl --judge-provider ollama --judge-model llama3.2

# İki koşuyu kıyasla (before → after)
openeval compare reports/report_before.json reports/report.json
```

---

## Dürüst sınır — "hakem de bir LLM, neden güveniyorsun?"
Tam güvenmiyoruz. Hakem bias'lı olabilir, kolay soruya cömert davranabilir. Azaltma
yolları: (a) judge'ı subject'ten güçlü seç, (b) `temperature=0` ile deterministik yap,
(c) `reasoning` tut ki her not denetlenebilir olsun, (d) çoklu hakem / insan spot-check
(bu aracın mevcut kapsamının ötesi). Bu sınırı bilmek, sistemi anlamanın parçasıdır.
