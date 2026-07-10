"""
Retriever based on ChromaDB + Ollama embeddings.
Replaced TF-IDF — semantic similarity is now used.
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
    A vector-based retriever using ChromaDB.

    On the first run:
      1. Loads the knowledge base
      2. Embeds each chunk (Ollama)
      3. Stores it in ChromaDB

    On subsequent runs:
      1. Loads from cache (does not re-embed)
      2. Embeds the query
      3. Returns the closest chunks
    """

    def __init__(self, top_k: int = 2, persist_dir: str = ".chromadb"):
        self.top_k = top_k
        self.embedder = OllamaEmbedder()

        # ChromaDB — persists to a local file, does not vanish when the app closes
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        """
        Load the collection if it exists, otherwise create and populate it.
        This way we do not re-embed every time.
        """
        existing = [c.name for c in self.client.list_collections()]

        if COLLECTION_NAME in existing:
            logger.info("ChromaDB collection found, loaded from cache")
            return self.client.get_collection(COLLECTION_NAME)

        logger.info("Creating ChromaDB collection, embedding the knowledge base...")
        collection = self.client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},  # use cosine similarity
        )

        self._index_documents(collection)
        return collection

    def _index_documents(self, collection):
        """Embed all chunks in the knowledge base and store them in ChromaDB."""
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

        logger.info("Knowledge base indexed: %d documents", len(documents))

    def retrieve(self, query: str) -> list[dict]:
        """
        Return the top_k chunks closest to the query.

        1. Embed the question
        2. Do a similarity search in ChromaDB
        3. Return the closest chunks
        """
        query_vector = self.embedder.embed(query)

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=self.top_k,
            include=["documents", "metadatas", "distances"],
        )

        output = []
        for i in range(len(results["ids"][0])):
            # ChromaDB returns cosine distance: 0=identical, 2=completely different
            # We want similarity: 1 - distance/2
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
        """Build a context string for the judge."""
        docs = self.retrieve(query)
        if not docs:
            return ""

        parts = []
        for doc in docs:
            parts.append(f"[{doc['topic']}] (similarity: {doc['score']})\n{doc['content']}")

        return "\n\n---\n\n".join(parts)

    def reset(self):
        """Delete and recreate the collection. Use this when the knowledge base changes."""
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self._get_or_create_collection()
        logger.info("Collection reset and recreated")
