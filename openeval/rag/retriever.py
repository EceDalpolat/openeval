"""
TF-IDF tabanli retriever.
Soruya gore knowledge base'den en alakali chunk'i bulur.
Hafta 3'te ChromaDB + embedding'e gecilecek.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from .knowledge_base import get_all_documents
from ..observability import get_logger

logger = get_logger(__name__)


class TFIDFRetriever:
    """
    TF-IDF ile keyword benzerligine dayali retriever.
    
    Nasil calisir:
    1. Tum dokumanlari vektorlestirir (TF-IDF matrix)
    2. Sorguyu ayni sekilde vektorlestirir
    3. Cosine similarity hesaplar
    4. En yuksek skorlu k dokumani dondurur
    """

    def __init__(self, top_k: int = 2):
        self.top_k = top_k
        self.documents = get_all_documents()
        self._build_index()

    def _build_index(self):
        """TF-IDF matrisini olustur — bir kez yapilir."""
        contents = [doc["content"] for doc in self.documents]
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),   # tek kelime + ikili kelime grupları
            max_features=5000,
            stop_words=None,      # Turkce/Ingilizce karisik, stop words kapatiyoruz
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(contents)
        logger.debug("TF-IDF index olusturuldu: %d dokuman", len(self.documents))

    def retrieve(self, query: str) -> list[dict]:
        """
        Sorguya en yakin top_k dokumani dondur.
        
        Returns:
            [{"topic": "RAG", "content": "...", "score": 0.85}, ...]
        """
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # En yuksek skorlu indexleri bul
        top_indices = np.argsort(similarities)[::-1][:self.top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score > 0.01:   # cok dusuk skoru atla
                results.append({
                    "topic": self.documents[idx]["topic"],
                    "content": self.documents[idx]["content"],
                    "score": round(score, 3),
                })
                logger.debug(
                    "Retrieved: topic=%s, score=%.3f",
                    self.documents[idx]["topic"], score
                )

        if not results:
            logger.warning("Hic alakali dokuman bulunamadi: query=%s", query[:50])

        return results

    def retrieve_as_context(self, query: str) -> str:
        """
        Retrieve sonuclarini judge icin tek string'e donustur.
        Bu string EvalCase.context alanina girecek.
        """
        docs = self.retrieve(query)
        if not docs:
            return ""

        parts = []
        for doc in docs:
            parts.append(f"[{doc['topic']}]\n{doc['content']}")

        return "\n\n---\n\n".join(parts)
