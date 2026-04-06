"""
Hybrid retriever with semantic search + BM25 + reranking.
"""

import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import CrossEncoder

from config import settings

reranker_model = CrossEncoder(settings.reranker_model)


def load_vectorstore() -> FAISS:
    vector_store_dir = Path(settings.vector_store_dir)

    if not vector_store_dir.exists():
        raise FileNotFoundError(
            f"Vector store directory not found: {vector_store_dir}. Run ingest.py first."
        )

    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.api_key.get_secret_value(),
    )

    return FAISS.load_local(
        str(vector_store_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def load_bm25_documents() -> list[Document]:
    chunks_path = Path(settings.vector_store_dir) / "chunks.json"

    if not chunks_path.exists():
        raise FileNotFoundError(
            f"BM25 chunks file not found: {chunks_path}. Run ingest.py first."
        )

    raw_chunks = json.loads(chunks_path.read_text(encoding="utf-8"))

    documents = [
        Document(
            page_content=item["page_content"],
            metadata=item.get("metadata", {}),
        )
        for item in raw_chunks
    ]
    return documents


def get_semantic_results(query: str) -> list[Document]:
    vectorstore = load_vectorstore()
    return vectorstore.similarity_search(query, k=settings.semantic_k)


def get_bm25_results(query: str) -> list[Document]:
    documents = load_bm25_documents()
    retriever = BM25Retriever.from_documents(documents)
    retriever.k = settings.bm25_k
    return retriever.invoke(query)


def deduplicate_documents(documents: list[Document]) -> list[Document]:
    seen = set()
    unique_docs = []

    for doc in documents:
        key = (
            doc.page_content.strip(),
            json.dumps(doc.metadata, sort_keys=True, ensure_ascii=False),
        )
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)

    return unique_docs


def rerank_documents(query: str, documents: list[Document]) -> list[Document]:
    unique_docs = deduplicate_documents(documents)

    if not unique_docs:
        return []

    pairs = [(query, doc.page_content) for doc in unique_docs]
    scores = reranker_model.predict(pairs)

    scored_docs = list(zip(unique_docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    reranked_docs = [doc for doc, _score in scored_docs[: settings.final_k]]
    return reranked_docs


def hybrid_search(query: str) -> list[Document]:
    semantic_docs = get_semantic_results(query)
    bm25_docs = get_bm25_results(query)

    combined_docs = semantic_docs + bm25_docs
    reranked_docs = rerank_documents(query, combined_docs)

    return reranked_docs


def format_documents(docs: list[Document]) -> str:
    if not docs:
        return "No relevant documents found."

    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "n/a")
        parts.append(
            f"[Document {i}] Source: {source}, Page: {page}\n{doc.page_content}"
        )

    return "\n\n".join(parts)


def search_knowledge_base(query: str) -> str:
    docs = hybrid_search(query)
    return format_documents(docs)