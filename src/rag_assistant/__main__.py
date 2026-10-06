"""Ingest local sample files without enabling the HTTP upload endpoint."""

import argparse
from pathlib import Path

from .config import Settings
from .documents import chunk_text, extract_text
from .embeddings import make_embedder
from .store import VectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest local text, Markdown, and text PDF files")
    parser.add_argument("command", choices=["ingest"])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--access-group", default="public")
    parser.add_argument("--chunk-size", type=int, default=600)
    parser.add_argument("--overlap", type=int, default=80)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    if not args.directory.is_dir():
        parser.error("directory does not exist")
    settings = Settings.from_env()
    if args.reset:
        settings.index_path.unlink(missing_ok=True)
    store = VectorStore(
        settings.index_path, make_embedder(settings.embedding_backend, settings.model)
    )
    count = 0
    for path in sorted(args.directory.iterdir()):
        if path.suffix.lower() not in {".txt", ".md", ".pdf"}:
            continue
        text = extract_text(path.name, path.read_bytes())
        count += store.add(
            path.name, args.access_group, chunk_text(text, args.chunk_size, args.overlap)
        )
    print(f"Ingested {count} chunks from {args.directory}")


if __name__ == "__main__":
    main()
