import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import chromadb
import duckdb
import ollama
import pymupdf
from dotenv import load_dotenv

CHAT_CONTEXT_TOKENS = 8192


def require_env(name):
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing {name} in the environment.")
    return value


def connect_services():
    load_dotenv()
    chunk_size = int(require_env("CHUNK_SIZE"))
    chunk_overlap = int(require_env("CHUNK_OVERLAP"))
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise RuntimeError("CHUNK_SIZE must be > CHUNK_OVERLAP >= 0.")

    chroma_persist_dir = require_env("CHROMA_PERSIST_DIR")
    duckdb_path = require_env("DUCKDB_PATH")
    pdf_dir = require_env("PDF_DIR")
    Path(chroma_persist_dir).mkdir(parents=True, exist_ok=True)
    Path(duckdb_path).parent.mkdir(parents=True, exist_ok=True)
    Path(pdf_dir).mkdir(parents=True, exist_ok=True)

    # DuckDB locks the file; one process per DUCKDB_PATH.
    duckdb_connection = duckdb.connect(duckdb_path)
    duckdb_connection.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            document_id VARCHAR PRIMARY KEY,
            source_path VARCHAR NOT NULL,
            original_name VARCHAR NOT NULL,
            ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    collection = chromadb.PersistentClient(path=chroma_persist_dir).get_or_create_collection(
        name=require_env("CHROMA_COLLECTION_NAME"),
        embedding_function=None,  # default MiniLM would mismatch nomic vectors
    )
    return SimpleNamespace(
        collection=collection,
        duckdb_connection=duckdb_connection,
        ollama_client=ollama.Client(host=require_env("OLLAMA_HOST")),
        pdf_dir=pdf_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=require_env("EMBEDDING_MODEL"),
        language_model=require_env("LLM_MODEL"),
        retrieve_count=int(require_env("RETRIEVE_COUNT")),
    )


def collect_pdf_paths(target):
    target_path = Path(target)
    if not target_path.exists():
        raise FileNotFoundError(f"Path does not exist: {target_path}")
    if target_path.is_file():
        if target_path.suffix.lower() != ".pdf":
            raise ValueError(f"Not a PDF: {target_path}")
        return [str(target_path.resolve())]
    if not target_path.is_dir():
        raise ValueError(f"Not a file or directory: {target_path}")
    pdf_paths = sorted(
        str(path.resolve())
        for path in target_path.iterdir()
        if path.is_file() and path.suffix.lower() == ".pdf"
    )
    if not pdf_paths:
        raise ValueError(f"No PDFs in {target_path}")
    return pdf_paths


def hash_file(path):
    with open(path, "rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def split_page_text(text, chunk_size, chunk_overlap):
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    step = chunk_size - chunk_overlap
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        if start + chunk_size >= len(text):
            break
        start += step
    return chunks


def chunks_from_pdf(path, chunk_size, chunk_overlap):
    chunks = []
    with pymupdf.open(path) as document:
        for page in document:
            for text in split_page_text(page.get_text(sort=True), chunk_size, chunk_overlap):
                chunks.append(
                    {
                        "text": text,
                        "page_number": page.number + 1,
                        "chunk_index": len(chunks),
                    }
                )
    return chunks


def delete_document(services, document_id=None, source_path=None):
    if source_path:
        services.duckdb_connection.execute(
            "DELETE FROM documents WHERE source_path = ?", [source_path]
        )
        services.collection.delete(where={"source_path": source_path})
        return
    if not document_id:
        return
    services.duckdb_connection.execute(
        "DELETE FROM documents WHERE document_id = ?", [document_id]
    )
    services.collection.delete(where={"document_id": document_id})


def ingest_pdf(services, path):
    document_id = hash_file(path)
    source_path = str(Path(path).resolve())
    already_in_duckdb = services.duckdb_connection.execute(
        "SELECT 1 FROM documents WHERE document_id = ?", [document_id]
    ).fetchone()
    already_in_chroma = bool(
        services.collection.get(where={"document_id": document_id}, limit=1)["ids"]
    )
    if already_in_duckdb and already_in_chroma:
        print(f"Skipping {path}, already ingested")
        return {
            "status": "skipped",
            "document_id": document_id,
            "source_path": source_path,
            "chunk_count": 0,
        }

    delete_document(services, document_id=document_id)
    delete_document(services, source_path=source_path)

    print(f"Ingesting {path}...")
    chunks = chunks_from_pdf(path, services.chunk_size, services.chunk_overlap)
    if not chunks:
        print(f"No extractable text in {path}, skipping")
        return {
            "status": "empty",
            "document_id": document_id,
            "source_path": source_path,
            "chunk_count": 0,
        }

    chunk_ids = [f"{document_id}:{chunk['chunk_index']}" for chunk in chunks]
    chunk_texts = [chunk["text"] for chunk in chunks]
    embeddings = services.ollama_client.embed(
        model=services.embedding_model, input=chunk_texts
    ).embeddings
    if len(embeddings) != len(chunk_texts):
        raise RuntimeError(f"Embedding count mismatch for {path}")

    try:
        services.collection.upsert(
            ids=chunk_ids,
            documents=chunk_texts,
            embeddings=list(embeddings),
            metadatas=[
                {
                    "document_id": document_id,
                    "source_path": source_path,
                    "page_number": chunk["page_number"],
                    "chunk_index": chunk["chunk_index"],
                }
                for chunk in chunks
            ],
        )
        services.duckdb_connection.execute(
            "INSERT INTO documents (document_id, source_path, original_name) VALUES (?, ?, ?)",
            [document_id, source_path, Path(path).name],
        )
    except Exception:
        delete_document(services, document_id=document_id)
        raise

    return {
        "status": "ingested",
        "document_id": document_id,
        "source_path": source_path,
        "chunk_count": len(chunks),
    }


def ingest_path(services, target):
    return [ingest_pdf(services, path) for path in collect_pdf_paths(target)]


def list_documents(services):
    rows = services.duckdb_connection.execute(
        """
        SELECT document_id, source_path, original_name, ingested_at
        FROM documents
        ORDER BY ingested_at DESC
        """
    ).fetchall()
    return [
        {
            "document_id": document_id,
            "source_path": source_path,
            "original_name": original_name,
            "ingested_at": str(ingested_at),
        }
        for document_id, source_path, original_name, ingested_at in rows
    ]


def get_document(services, document_id):
    row = services.duckdb_connection.execute(
        """
        SELECT document_id, source_path, original_name, ingested_at
        FROM documents
        WHERE document_id = ?
        """,
        [document_id],
    ).fetchone()
    if not row:
        return None
    document_id, source_path, original_name, ingested_at = row
    stored = services.collection.get(
        where={"document_id": document_id},
        include=["documents", "metadatas"],
    )
    chunks = []
    for chunk_id, text, metadata in zip(
        stored["ids"], stored["documents"], stored["metadatas"]
    ):
        chunks.append(
            {
                "chunk_id": chunk_id,
                "page_number": metadata.get("page_number"),
                "chunk_index": metadata.get("chunk_index"),
                "content_length": len(text),
            }
        )
    chunks.sort(key=lambda chunk: chunk["chunk_index"] or 0)
    return {
        "document_id": document_id,
        "source_path": source_path,
        "original_name": original_name,
        "ingested_at": str(ingested_at),
        "chunks": chunks,
    }


def retrieved_chunks(services, question):
    match_count = min(services.retrieve_count, services.collection.count())
    if match_count < 1:
        return []
    question_embedding = services.ollama_client.embed(
        model=services.embedding_model, input=question
    ).embeddings[0]
    results = services.collection.query(
        query_embeddings=[question_embedding],
        n_results=match_count,
        include=["documents", "metadatas"],
    )
    texts = results["documents"][0]
    metadatas = results["metadatas"][0]
    if not texts:
        return []
    return [
        {
            "text": text,
            "source_path": metadata.get("source_path", "unknown"),
            "page_number": metadata.get("page_number", "?"),
        }
        for text, metadata in zip(texts, metadatas)
    ]


def chat_messages(chunks, question):
    excerpts = "\n\n".join(
        f"[{chunk['source_path']} p.{chunk['page_number']}]\n{chunk['text']}"
        for chunk in chunks
    )
    return [
        {
            "role": "system",
            "content": (
                "Answer only from the excerpts. "
                "After a claim, copy the excerpt's bracketed header, including the file path and page. "
                "If the excerpts are not enough, say you do not know."
            ),
        },
        {
            "role": "user",
            "content": f"Excerpts:\n\n{excerpts}\n\nQuestion: {question}",
        },
    ]


def answer_question(services, question):
    chunks = retrieved_chunks(services, question)
    if not chunks:
        return {"answer": None, "citations": []}
    response = services.ollama_client.chat(
        model=services.language_model,
        messages=chat_messages(chunks, question),
        options={"num_ctx": CHAT_CONTEXT_TOKENS},
    )
    return {
        "answer": response.message.content,
        "citations": [
            {
                "source_path": chunk["source_path"],
                "page_number": chunk["page_number"],
                "text": chunk["text"],
            }
            for chunk in chunks
        ],
    }
