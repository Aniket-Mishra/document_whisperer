# document_whisperer

Talk to PDFs on this machine. Ingest a file or a top-level folder, then ask. Ollama runs `nomic-embed-text` and `mistral-nemo:12b`. FastAPI is the HTTP app. DuckDB is the document catalog. Chroma holds the chunks.

## Setup

```sh
ollama serve
ollama pull nomic-embed-text
ollama pull mistral-nemo:12b

uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

Settings live in `.env`.

## Run the API

```sh
python api.py
```

Open http://localhost:8000/docs to upload a PDF and query it.

```sh
curl -F file=@paper.pdf http://localhost:8000/ingest/upload
curl -X POST http://localhost:8000/ingest/path \
  -H 'Content-Type: application/json' \
  -d '{"path":"data/pdfs"}'
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is this paper about?"}'
curl http://localhost:8000/documents
```

`POST /ingest/path` reads a PDF, or every `*.pdf` in a folder (not nested). Uploads land in `PDF_DIR` (`data/pdfs`). Keep `data/` with the process. One process per DuckDB file.

## CLI

```sh
python cli.py path/to/paper.pdf
python cli.py path/to/folder
```

Type a question. Type quit to exit.
