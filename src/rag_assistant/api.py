"""FastAPI endpoints for document ingestion and cited question answering."""

import logging
import re
import secrets
from pathlib import Path
from time import perf_counter
from typing import Annotated

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .config import Settings
from .documents import MAX_UPLOAD_BYTES, chunk_text, extract_text
from .embeddings import make_embedder
from .generation import generate
from .store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    top_k: int = Field(default=3, ge=1, le=5)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    store = VectorStore(
        settings.index_path,
        make_embedder(settings.embedding_backend, settings.model),
    )
    app = FastAPI(title="Enterprise RAG Assistant", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "chunks": len(store.rows), "embedding": store.embedder.name}

    @app.post("/documents", status_code=201)
    async def ingest(
        file: Annotated[UploadFile, File()],
        access_group: Annotated[str, Form()] = "public",
        chunk_size: Annotated[int, Form()] = 600,
        overlap: Annotated[int, Form()] = 80,
        x_admin_key: Annotated[str | None, Header()] = None,
    ) -> dict:
        if not settings.admin_key:
            raise HTTPException(503, "API ingestion is disabled; configure RAG_ADMIN_KEY")
        if not x_admin_key or not secrets.compare_digest(x_admin_key, settings.admin_key):
            raise HTTPException(403, "Admin key required")
        if not re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", access_group):
            raise HTTPException(422, "Invalid access group")
        try:
            content = await file.read(MAX_UPLOAD_BYTES + 1)
            title = Path(file.filename or "upload").name
            text = extract_text(title, content)
            chunks = chunk_text(text, chunk_size, overlap)
            count = store.add(title, access_group, chunks)
        except (ValueError, UnicodeError) as exc:
            raise HTTPException(422, str(exc)) from exc
        logger.info("Ingested %s chunks for document %s", count, title)
        return {"title": title, "access_group": access_group, "chunks": count}

    @app.post("/query")
    async def query(
        request: QueryRequest, x_access_token: str | None = Header(default=None)
    ) -> dict:
        start = perf_counter()
        groups = settings.allowed_groups(x_access_token)
        hits = store.search(request.question, groups, request.top_k)
        try:
            answer, citations, usage = await generate(request.question, hits, settings)
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc
        cited = [hit for hit in hits if hit.source_id in citations]
        elapsed = round((perf_counter() - start) * 1000, 2)
        logger.info(
            "Query completed hits=%s citations=%s latency_ms=%s", len(hits), len(cited), elapsed
        )
        return {
            "answer": answer,
            "mode": settings.generator,
            "citations": [
                {"source_id": hit.source_id, "title": hit.title, "excerpt": hit.text[:360]}
                for hit in cited
            ],
            "retrieved": len(hits),
            "latency_ms": elapsed,
            "usage": usage,
        }

    return app


app = create_app()
