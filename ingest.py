"""
Knowledge ingestion pipeline.

Loads documents from data/ directory, splits into chunks,
generates embeddings, and saves the index to disk.

Usage: python ingest.py
"""

import json
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from config import settings


def load_documents() -> list[Document]:
    data_path = Path(settings.data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    documents: list[Document] = []

    for pdf_file in data_path.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf_file))
        documents.extend(loader.load())

    for txt_file in data_path.glob("*.txt"):
        loader = TextLoader(str(txt_file), encoding="utf-8")
        documents.extend(loader.load())

    for md_file in data_path.glob("*.md"):
        loader = TextLoader(str(md_file), encoding="utf-8")
        documents.extend(loader.load())

    if not documents:
        raise ValueError(f"No PDF, TXT, or MD files found in {data_path}")

    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return splitter.split_documents(documents)


def save_bm25_chunks(chunks: list[Document], output_path: Path) -> None:
    serialized_chunks = []

    for chunk in chunks:
        serialized_chunks.append(
            {
                "page_content": chunk.page_content,
                "metadata": chunk.metadata,
            }
        )

    output_path.write_text(
        json.dumps(serialized_chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def ingest() -> None:
    documents = load_documents()
    chunks = split_documents(documents)

    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.api_key.get_secret_value(),
    )

    vector_store_dir = Path(settings.vector_store_dir)
    vector_store_dir.mkdir(parents=True, exist_ok=True)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(vector_store_dir))

    bm25_chunks_path = vector_store_dir / "chunks.json"
    save_bm25_chunks(chunks, bm25_chunks_path)

    print(f"Loaded documents: {len(documents)}")
    print(f"Created chunks: {len(chunks)}")
    print(f"Vector index saved to: {vector_store_dir}")
    print(f"BM25 chunks saved to: {bm25_chunks_path}")


if __name__ == "__main__":
    ingest()