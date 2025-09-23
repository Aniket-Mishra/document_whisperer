import os
import hashlib
import duckdb
import chromadb
import ollama
import unstructured
from unstructured.partition.pdf import partition_pdf
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

PDF_DIR = os.getenv("PDF_DIR")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME")
DUCKDB_PATH = os.getenv("DUCKDB_PATH")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))

logger.add("ingestion.log", rotation="15 MB", level="INFO")
chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)
con = duckdb.connect(DUCKDB_PATH)

def initialize_db():
    """Create DuckDB tables if they don't exist."""
    con.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id VARCHAR PRIMARY KEY, file_path VARCHAR NOT NULL, title VARCHAR,
            authors VARCHAR, venue VARCHAR, year INTEGER, doi VARCHAR, bibtex TEXT,
            ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id VARCHAR PRIMARY KEY, doc_id VARCHAR REFERENCES documents(doc_id),
            page_num INTEGER, section_title VARCHAR, content_length INTEGER,
            content_hash VARCHAR
        );
    """)
    logger.info("DuckDB tables initialized.")

def get_doc_hash(file_path):
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()

def is_processed(doc_id):
    """Check if a document has already been processed."""
    res = con.execute("SELECT COUNT(*) FROM documents WHERE doc_id = ?", [doc_id]).fetchone()
    return res[0] > 0

def extract_elements(file_path):
    """Extract structured elements from a PDF using unstructured."""
    logger.info(f"Parsing PDF: {os.path.basename(file_path)}")
    return partition_pdf(
        file_path,
        # Using "hi_res" strategy for better layout detection, suitable for research papers
        strategy="hi_res",
        # We can add more metadata extraction rules here if needed
        extract_image_block_types=[], # Ignore images for now
        infer_table_structure=False, # Ignore tables for now
    )

def chunk_elements(elements):
    """Chunk document elements based on headings (sections)."""
    chunks = []
    current_section_title = "Introduction" # Default for text before first heading
    current_chunk = ""

    for el in elements:
        if isinstance(el, unstructured.documents.elements.Title):
            if el.text.lower() in ["abstract", "references"]: # Skip abstract and refs for now
                continue
            if len(current_chunk) > CHUNK_OVERLAP: # Save previous chunk
                 chunks.append({"text": current_chunk.strip(), "section": current_section_title, "page": el.metadata.page_number})
            current_section_title = el.text
            current_chunk = "" # Start a new chunk
        elif isinstance(el, unstructured.documents.elements.NarrativeText) or isinstance(el, unstructured.documents.elements.ListItem):
            current_chunk += "\n" + el.text

    if current_chunk: # Add the last chunk
        chunks.append({"text": current_chunk.strip(), "section": current_section_title, "page": elements[-1].metadata.page_number})

    # Further split large chunks if they exceed CHUNK_SIZE
    final_chunks = []
    for chunk in chunks:
        if len(chunk["text"]) > CHUNK_SIZE:
            # Simple recursive split for now, can be improved
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            sub_chunks = text_splitter.split_text(chunk["text"])
            for sub_chunk in sub_chunks:
                final_chunks.append({"text": sub_chunk, "section": chunk["section"], "page": chunk["page"]})
        else:
            final_chunks.append(chunk)

    logger.info(f"Created {len(final_chunks)} chunks.")
    return final_chunks

def process_pdf(file_path):
    """Main processing function for a single PDF."""
    doc_id = get_doc_hash(file_path)
    if is_processed(doc_id):
        logger.info(f"Skipping already processed document: {os.path.basename(file_path)} (ID: {doc_id})")
        return

    elements = extract_elements(file_path)

    # Simplified metadata extraction
    title = elements[0].text if elements and isinstance(elements[0], unstructured.documents.elements.Title) else "Unknown Title"
    authors = "Unknown Authors" # Placeholder, more robust extraction is complex
    year = 2024 # Placeholder

    # Store document metadata in DuckDB
    con.execute("INSERT INTO documents (doc_id, file_path, title, authors, year) VALUES (?, ?, ?, ?, ?)",
                [doc_id, file_path, title, authors, year])
    logger.info(f"Stored document metadata for '{title}'")

    chunks_data = chunk_elements(elements)
    if not chunks_data:
        logger.warning(f"No chunks were generated for {file_path}")
        return

    # Prepare for batch insertion
    chunk_ids = []
    chunk_texts = []
    chunk_metadatas = []
    duckdb_chunk_records = []

    for i, chunk in enumerate(chunks_data):
        chunk_id = f"{doc_id}_chunk_{i}"
        content_hash = hashlib.sha256(chunk["text"].encode()).hexdigest()

        chunk_ids.append(chunk_id)
        chunk_texts.append(chunk["text"])
        chunk_metadatas.append({
            "doc_id": doc_id,
            "paper_title": title,
            "authors": authors,
            "year": year,
            "section_title": chunk["section"],
            "page_num": chunk["page"],
            "chunk_id": chunk_id,
        })
        duckdb_chunk_records.append((chunk_id, doc_id, chunk["page"], chunk["section"], len(chunk["text"]), content_hash))

    # Batch embed and insert into ChromaDB
    logger.info(f"Generating embeddings for {len(chunk_texts)} chunks...")
    embeddings = [
        res["embedding"]
        for res in ollama.embed(model=EMBEDDING_MODEL, prompts=chunk_texts)["embeddings"]
    ]

    collection.add(
        ids=chunk_ids,
        embeddings=embeddings,
        documents=chunk_texts,
        metadatas=chunk_metadatas
    )
    logger.info("Added chunks to ChromaDB.")

    # Batch insert into DuckDB
    con.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)", duckdb_chunk_records)
    logger.info("Added chunk metadata to DuckDB.")


def main():
    """Main function to run the ingestion pipeline."""
    initialize_db()
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
    logger.info(f"Found {len(pdf_files)} PDFs in '{PDF_DIR}'")

    for pdf_file in pdf_files:
        file_path = os.path.join(PDF_DIR, pdf_file)
        try:
            process_pdf(file_path)
        except Exception as e:
            logger.error(f"Failed to process {pdf_file}: {e}")

    con.close()
    logger.info("Ingestion complete.")

if __name__ == "__main__":
    main()