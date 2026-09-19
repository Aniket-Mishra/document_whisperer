from contextlib import asynccontextmanager
from pathlib import Path

import ollama
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from document_whisperer import (
    answer_question,
    connect_services,
    get_document,
    ingest_path,
    ingest_pdf,
    list_documents,
)


class PathIngestRequest(BaseModel):
    path: str


class QueryRequest(BaseModel):
    question: str


def raise_from_backend(error):
    if isinstance(error, ollama.ResponseError):
        detail = str(error)
        if error.status_code == 404:
            detail = f"{error}. Pull the missing model with ollama pull."
        raise HTTPException(status_code=502, detail=detail) from error
    raise HTTPException(status_code=500, detail=str(error)) from error


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.services = connect_services()
    yield


app = FastAPI(title="document_whisperer", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return RedirectResponse("/docs")


@app.get("/health")
def health():
    try:
        app.state.services.ollama_client.list()
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail="Ollama is not reachable. Start it with `ollama serve`.",
        ) from error
    return {"ok": True}


@app.post("/ingest/path")
def ingest_from_path(request: PathIngestRequest):
    try:
        results = ingest_path(app.state.services, request.path)
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise_from_backend(error)
    return {"results": results}


@app.post("/ingest/upload")
async def ingest_upload(file: UploadFile = File(...)):
    filename = Path(file.filename or "").name
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Not a PDF")
    destination = Path(app.state.services.pdf_dir) / filename
    destination.write_bytes(await file.read())
    try:
        return ingest_pdf(app.state.services, str(destination))
    except Exception as error:
        raise_from_backend(error)


@app.post("/query")
def query(request: QueryRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is empty")
    try:
        return answer_question(app.state.services, question)
    except Exception as error:
        raise_from_backend(error)


@app.get("/documents")
def documents():
    return {"documents": list_documents(app.state.services)}


@app.get("/documents/{document_id}")
def document(document_id: str):
    found = get_document(app.state.services, document_id)
    if found is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return found


if __name__ == "__main__":
    import os

    import uvicorn
    from dotenv import load_dotenv

    load_dotenv()
    uvicorn.run(
        app,
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
    )
