"""
ChromaDB + Ollama embedding tabanli retriever.
TF-IDF'in yerini aldi — artik anlam benzerligi kullaniliyor.
"""
import chromadb
from chromadb.config import Settings
from .knowledge_base import get_all_documents
from .embedder import OllamaEmbedder
from ..observability import get_logger

logger = get_logger(__name__)

COLLECTION_NAME = "openeval_knowledge"


class ChromaRetriever:
    """
    ChromaDB ile vektör tabanlı retriever.
    
    Ilk calistirildiginda:
      1. Knowledge base'i yukler
      2. Her chunk'i embed eder (Ollama)
      3. ChromaDB'ye kaydeder
    
    Sonraki calismalarda:
      1. Cache'den yukler (tekrar embed etmez)
      2. Sorguyu embed eder
      3. En yakin chunk'lari dondurur
    """

    def __init__(self, top_k: int = 2, persist_dir: str = ".chromadb"):
        self.top_k = top_k
        self.embedder = OllamaEmbedder()

        # ChromaDB — lokal dosyaya kaydeder, uygulama kapaninca kaybolmaz
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        """
        Collection varsa yukle, yoksa olustur ve doldur.
        Bu sayede her seferinde tekrar embed etmiyoruz.
        """
        existing = [c.name for c in self.client.list_collections()]

        if COLLECTION_NAME in existing:
            logger.info("ChromaDB collection bulundu, cache'den yuklendi")
            return self.client.get_collection(COLLECTION_NAME)

        logger.info("ChromaDB collection olusturuluyor, knowledge base embed ediliyor...")
        collection = self.client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},  # cosine similarity kullan
        )

        self._index_documents(collection)
        return collection

    def _index_documents(self, collection):
        """Knowledge base'deki tum chunk'lari embed edip ChromaDB'ye kaydet."""
        documents = get_all_documents()

        for doc in documents:
            vector = self.embedder.embed(doc["content"])
            collection.add(
                ids=[doc["id"]],
                embeddings=[vector],
                documents=[doc["content"]],
                metadatas=[{"topic": doc["topic"]}],
            )
            logger.info("Indexed: %s", doc["topic"])

        logger.info("Knowledge base indexlendi: %d dokuman", len(documents))

    def retrieve(self, query: str) -> list[dict]:
        """
        Sorguya en yakin top_k chunk'i dondur.
        
        1. Soruyu embed et
        2. ChromaDB'de similarity search yap
        3. En yakin chunk'lari dondur
        """
        query_vector = self.embedder.embed(query)

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=self.top_k,
            include=["documents", "metadatas", "distances"],
        )

        output = []
        for i in range(len(results["ids"][0])):
            # ChromaDB cosine distance dondurur: 0=ayni, 2=tamamen farkli
            # Biz similarity istiyoruz: 1 - distance/2
            distance = results["distances"][0][i]
            similarity = round(1 - distance / 2, 3)

            output.append({
                "topic": results["metadatas"][0][i]["topic"],
                "content": results["documents"][0][i],
                "score": similarity,
            })
            logger.debug(
                "Retrieved: topic=%s, similarity=%.3f",
                results["metadatas"][0][i]["topic"],
                similarity,
            )

        return output

    def retrieve_as_context(self, query: str) -> str:
        """Judge icin context string olustur."""
        docs = self.retrieve(query)
        if not docs:
            return ""

        parts = []
        for doc in docs:
            parts.append(f"[{doc['topic']}] (similarity: {doc['score']})\n{doc['content']}")

        return "\n\n---\n\n".join(parts)

    def reset(self):
        """Collection'i sil ve yeniden olustur. Knowledge base degisince kullan."""
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self._get_or_create_collection()
        logger.info("Collection sifirlanip yeniden olusturuldu")
