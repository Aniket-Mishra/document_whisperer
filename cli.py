import sys

import ollama

from document_whisperer import (
    CHAT_CONTEXT_TOKENS,
    chat_messages,
    connect_services,
    ingest_path,
    retrieved_chunks,
)


def stream_answer(services, question):
    chunks = retrieved_chunks(services, question)
    if not chunks:
        print("Nothing retrieved for that question.")
        return
    stream = services.ollama_client.chat(
        model=services.language_model,
        messages=chat_messages(chunks, question),
        stream=True,
        options={"num_ctx": CHAT_CONTEXT_TOKENS},
    )
    for chunk in stream:
        print(chunk.message.content, end="", flush=True)
    print()


def run_chat(services):
    print("Chat is ready. Type quit to exit.")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if question.lower() in {"quit", "exit"}:
            return
        if not question:
            continue
        stream_answer(services, question)


def main():
    if len(sys.argv) != 2:
        print("Usage: python cli.py <pdf-file-or-directory>")
        sys.exit(1)

    try:
        services = connect_services()
        services.ollama_client.list()
    except Exception as error:
        print(error)
        print("Start Ollama with `ollama serve`, then pull EMBEDDING_MODEL and LLM_MODEL.")
        sys.exit(1)

    try:
        ingest_path(services, sys.argv[1])
    except (FileNotFoundError, ValueError, ollama.ResponseError) as error:
        print(error)
        sys.exit(1)

    if services.collection.count() < 1:
        print("Nothing is searchable.")
        sys.exit(1)

    try:
        run_chat(services)
    except ollama.ResponseError as error:
        print(error)
        sys.exit(1)


if __name__ == "__main__":
    main()
