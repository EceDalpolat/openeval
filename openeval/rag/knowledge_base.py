"""
AI/LLM kavramlari knowledge base.
Judge bu bilgilere bakarak degerlendirme yapar.
"""

DOCUMENTS = [
    {
        "id": "rag",
        "topic": "RAG",
        "content": """RAG (Retrieval-Augmented Generation), LLM'e harici bilgi kaynagi ekleyen bir tekniktir.
Model egitim verisinde olmayan guncel bilgileri sorgu aninda alabilir.
Calisma sekli: 1) Belgeler embedding'e cevirilir ve vektorDB'ye kaydedilir.
2) Kullanici sorusu gelince benzer belgeler retrieve edilir.
3) Bu belgeler + soru birlikte LLM'e gonderilir.
4) LLM kaynaga bakarak cevap uretir.
RAG vs Fine-tuning: RAG dinamik/guncel bilgiler icin, fine-tuning model davranisini degistirmek icin kullanilir.
Kaynak gosterme RAG'in en buyuk avantajidir — model nereden bildigini soyleyebilir."""
    },
    {
        "id": "embedding",
        "topic": "Embedding",
        "content": """Embedding, metni sayisal vektore donusturmektir.
Benzer anlamli metinler vektör uzayinda birbirine yakin durur.
Ornek: 'kedi' ve 'kopek' vektorleri birbirine yakinken 'ucak' uzakta durur.
Boyut: tipik embedding modeli 768-3072 boyutlu vektor uretir.
Kullanim alanlari: semantic search, oneri sistemleri, RAG, duplicate detection.
OpenAI text-embedding-3-small ve text-embedding-3-large populer modellerdir.
Cosine similarity ile iki vektorun benzerligi olculur: 0.0=tamamen farkli, 1.0=ayni anlam."""
    },
    {
        "id": "fine_tuning",
        "topic": "Fine-tuning",
        "content": """Fine-tuning, hazir bir modeli kendi verin uzerinde yeniden egitmektir.
Base model genel bilgiye sahiptir, fine-tuning onu belirli bir goreve adapte eder.
LoRA (Low-Rank Adaptation): modelin tum parametrelerini degil, kucuk adapter katmanlarini egitir.
QLoRA: LoRA + quantization, daha az GPU RAM gerektirir, M4 Mac gibi consumer donanim icin uygundur.
Ne zaman kullanilir: belirli ton/stil ogretmek, domain-specific bilgi, hiz optimizasyonu.
Ne zaman kullanilmaz: guncel bilgi icin (RAG daha iyi), kucuk dataset ile (overfitting riski).
Hugging Face PEFT kutuphanesi LoRA/QLoRA icin standart aractir."""
    },
    {
        "id": "llm_evaluation",
        "topic": "LLM Evaluation",
        "content": """LLM degerlendirmesi, model ciktilarinin kalitesini olcmektir.
Temel boyutlar: faithfulness (gercekle uyum), relevance (soruyla ilgi),
clarity (anlasilirlik), safety (guvenlik), consistency (tutarlilik).
LLM-as-judge: guclu bir modelin (GPT-4, Claude) diger modeli degerlendirmesi.
Human evaluation altın standarttır ama pahali ve yavas.
RAGAS: RAG sistemleri icin ozel eval framework.
Hallucination: modelin yanlis ama emin gorunen cevap uretmesi — en kritik sorun.
Benchmark: MMLU, HumanEval, HellaSwag standart degerlendirme veri setleridir."""
    },
    {
        "id": "llm_basics",
        "topic": "LLM Temelleri",
        "content": """LLM (Large Language Model), milyarlarca parametre ile egitilmis dil modelidir.
Token: modelin islettigi en kucuk birim, kelime veya hece parcasidir.
Context window: modelin tek seferde isleyebildigi maksimum token sayisi.
Temperature: cevabın ne kadar yaratici/rastgele olacagini belirler. 0=deterministik, 1=yaratici.
Inference: modelin egitildikten sonra cevap urettigi sure.
Hallucination: modelin yanlis ama emin gorunen bilgi uretmesi.
GPT-4o, Claude, Gemini kapali kaynak; Llama, Mistral acik kaynak modellerdir.
Prompt engineering: modelden istenen ciktiyi almak icin girdi tasarlama sanatidir."""
    },
    {
        "id": "vector_db",
        "topic": "Vektor Veritabani",
        "content": """Vektor veritabanlari, embedding vektorlerini saklar ve hizli similarity search yapar.
Normal SQL veritabanlari vektör aramasina uygun degildir.
ANN (Approximate Nearest Neighbor): tam eslesme yerine en yakin vektorleri bulur.
Populer secenekler: Pinecone (bulut, yonetilen), Chroma (lokal, basit),
Qdrant (Rust, performansli), FAISS (Meta, buyuk olcek), Weaviate (GraphQL destekli).
Indexing: vektorleri hizli aranabilir sekilde saklamak.
Metadata filtering: vektör arama + keyword filtreyi birlikte kullanmak."""
    },
    {
        "id": "agents",
        "topic": "AI Agents",
        "content": """AI Agent, sadece cevap uretmekle kalmayip aksiyon alabilen LLM sistemidir.
ReAct pattern: Reason (dusun) + Act (aksiyon al) dongusu.
Tool use: model web arama, kod calistirma, API cagrisi yapabilir.
LangGraph: agent workflow'lari icin graph-based framework.
Multi-agent: birden fazla ajanin birlikte calistigi sistem.
Ornek: 'Bu kodu yaz, test et, hatalari duzelt' — model tum adimlari atar.
Hafiza: agent onceki konusmalari hatirlamak icin memory mekanizmasi kullanir."""
    },
    {
        "id": "prompt_engineering",
        "topic": "Prompt Engineering",
        "content": """Prompt engineering, modelden istenen ciktiyi almak icin girdi tasarlamaktir.
Zero-shot: hic ornek vermeden direkt soru.
Few-shot: birkas ornek vererek modeli yonlendirme.
Chain of Thought (CoT): 'adim adim dusun' diyerek akil yurutmeyi zorlama.
System prompt: modelin genel davranisini belirleyen talimat.
Temperature 0: deterministik, tutarli cikti — eval icin ideal.
Negative prompting: 'su yanlisligi yapma' ile istenmeyen davranisi onleme.
Structured output: JSON mode ile modelin belirli formatta cevap vermesi."""
    },
]

def get_all_documents() -> list[dict]:
    return DOCUMENTS

def get_document_by_topic(topic: str) -> dict | None:
    for doc in DOCUMENTS:
        if doc["topic"].lower() == topic.lower():
            return doc
    return None
