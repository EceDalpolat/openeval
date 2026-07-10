"""
Knowledge base of AI/LLM concepts.
The judge draws on this information when evaluating.
"""

DOCUMENTS = [
    {
        "id": "rag",
        "topic": "RAG",
        "content": """RAG (Retrieval-Augmented Generation) is a technique that adds an external knowledge source to an LLM.
The model can fetch up-to-date information not in its training data at query time.
How it works: 1) Documents are converted to embeddings and stored in a vector DB.
2) When a user question arrives, similar documents are retrieved.
3) These documents + the question are sent to the LLM together.
4) The LLM produces an answer based on the source.
RAG vs Fine-tuning: RAG is for dynamic/up-to-date information, fine-tuning is for changing model behavior.
Citing sources is RAG's biggest advantage — the model can say where it learned something."""
    },
    {
        "id": "embedding",
        "topic": "Embedding",
        "content": """Embedding is the process of converting text into a numeric vector.
Semantically similar texts end up close to each other in vector space.
Example: the vectors for 'cat' and 'dog' are close, while 'airplane' is far away.
Dimensionality: a typical embedding model produces 768-3072 dimensional vectors.
Use cases: semantic search, recommendation systems, RAG, duplicate detection.
OpenAI's text-embedding-3-small and text-embedding-3-large are popular models.
Cosine similarity measures how similar two vectors are: 0.0=completely different, 1.0=same meaning."""
    },
    {
        "id": "fine_tuning",
        "topic": "Fine-tuning",
        "content": """Fine-tuning is retraining an existing model on your own data.
A base model has general knowledge; fine-tuning adapts it to a specific task.
LoRA (Low-Rank Adaptation): trains small adapter layers instead of all of the model's parameters.
QLoRA: LoRA + quantization, requires less GPU RAM, suitable for consumer hardware like an M4 Mac.
When to use it: teaching a specific tone/style, domain-specific knowledge, speed optimization.
When not to use it: for up-to-date information (RAG is better), with a small dataset (overfitting risk).
The Hugging Face PEFT library is the standard tool for LoRA/QLoRA."""
    },
    {
        "id": "llm_evaluation",
        "topic": "LLM Evaluation",
        "content": """LLM evaluation is measuring the quality of model outputs.
Core dimensions: faithfulness (agreement with facts), relevance (relevance to the question),
clarity (understandability), safety (safety), consistency (internal consistency).
LLM-as-judge: a strong model (GPT-4, Claude) evaluating another model.
Human evaluation is the gold standard but expensive and slow.
RAGAS: a dedicated eval framework for RAG systems.
Hallucination: the model producing a wrong but confident-looking answer — the most critical problem.
Benchmarks: MMLU, HumanEval, HellaSwag are standard evaluation datasets."""
    },
    {
        "id": "llm_basics",
        "topic": "LLM Basics",
        "content": """An LLM (Large Language Model) is a language model trained with billions of parameters.
Token: the smallest unit the model processes, a word or part of a word.
Context window: the maximum number of tokens the model can process at once.
Temperature: controls how creative/random the answer is. 0=deterministic, 1=creative.
Inference: the phase where the model produces answers after being trained.
Hallucination: the model producing wrong but confident-looking information.
GPT-4o, Claude, Gemini are closed source; Llama, Mistral are open source models.
Prompt engineering: the art of designing input to get the desired output from the model."""
    },
    {
        "id": "vector_db",
        "topic": "Vector Database",
        "content": """Vector databases store embedding vectors and do fast similarity search.
Ordinary SQL databases are not suited to vector search.
ANN (Approximate Nearest Neighbor): finds the closest vectors instead of an exact match.
Popular options: Pinecone (cloud, managed), Chroma (local, simple),
Qdrant (Rust, performant), FAISS (Meta, large scale), Weaviate (GraphQL support).
Indexing: storing vectors so they can be searched quickly.
Metadata filtering: combining vector search with keyword filters."""
    },
    {
        "id": "agents",
        "topic": "AI Agents",
        "content": """An AI Agent is an LLM system that can take actions, not just produce answers.
ReAct pattern: a Reason (think) + Act (take action) loop.
Tool use: the model can search the web, run code, and call APIs.
LangGraph: a graph-based framework for agent workflows.
Multi-agent: a system where multiple agents work together.
Example: 'Write this code, test it, fix the errors' — the model handles all the steps.
Memory: an agent uses a memory mechanism to remember previous conversations."""
    },
    {
        "id": "prompt_engineering",
        "topic": "Prompt Engineering",
        "content": """Prompt engineering is designing input to get the desired output from the model.
Zero-shot: asking a question directly without any examples.
Few-shot: guiding the model by giving a few examples.
Chain of Thought (CoT): forcing reasoning by saying 'think step by step'.
System prompt: the instruction that sets the model's overall behavior.
Temperature 0: deterministic, consistent output — ideal for evaluation.
Negative prompting: preventing unwanted behavior with 'do not do this'.
Structured output: having the model answer in a specific format via JSON mode."""
    },
]

def get_all_documents() -> list[dict]:
    return DOCUMENTS

def get_document_by_topic(topic: str) -> dict | None:
    for doc in DOCUMENTS:
        if doc["topic"].lower() == topic.lower():
            return doc
    return None
