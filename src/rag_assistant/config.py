"""Environment configuration and access policy."""

import json
import os
import secrets
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    index_path: Path = Path("index.json")
    embedding_backend: str = "hash"
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    generator: str = "extractive"
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.2:3b"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    openai_api_key: str = ""
    admin_key: str = ""
    read_tokens: dict[str, list[str]] | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        raw = os.getenv("RAG_READ_TOKENS_JSON", "{}")
        tokens = json.loads(raw)
        if not isinstance(tokens, dict) or any(
            not isinstance(k, str)
            or not isinstance(v, list)
            or any(not isinstance(group, str) for group in v)
            for k, v in tokens.items()
        ):
            raise ValueError("RAG_READ_TOKENS_JSON must map tokens to group lists")
        return cls(
            index_path=Path(os.getenv("RAG_INDEX_PATH", "index.json")),
            embedding_backend=os.getenv("RAG_EMBEDDING_BACKEND", "hash"),
            model=os.getenv("RAG_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            generator=os.getenv("RAG_GENERATOR", "extractive"),
            ollama_url=os.getenv("RAG_OLLAMA_URL", "http://127.0.0.1:11434"),
            ollama_model=os.getenv("RAG_OLLAMA_MODEL", "llama3.2:3b"),
            openai_base_url=os.getenv("RAG_OPENAI_BASE_URL", "https://api.openai.com/v1"),
            openai_model=os.getenv("RAG_OPENAI_MODEL", "gpt-4o-mini"),
            openai_api_key=os.getenv("RAG_OPENAI_API_KEY", ""),
            admin_key=os.getenv("RAG_ADMIN_KEY", ""),
            read_tokens=tokens,
        )

    def allowed_groups(self, token: str | None) -> set[str]:
        groups = {"public"}
        for known, assigned in (self.read_tokens or {}).items():
            if token and secrets.compare_digest(token, known):
                groups.update(assigned)
        return groups
