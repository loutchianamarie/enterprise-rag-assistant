from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag_assistant.api import create_app
from rag_assistant.config import Settings
from rag_assistant.documents import chunk_text, extract_text


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    settings = Settings(
        index_path=tmp_path / "index.json",
        embedding_backend="hash",
        generator="extractive",
        admin_key="test-admin-key",
        read_tokens={"test-reader-token": ["public", "team"]},
    )
    return TestClient(create_app(settings))


def upload(client: TestClient, name: str, text: str, group: str = "public"):
    return client.post(
        "/documents",
        headers={"X-Admin-Key": "test-admin-key"},
        data={"access_group": group},
        files={"file": (name, text.encode(), "text/plain")},
    )


def test_cited_answer_and_persisted_index(client: TestClient, tmp_path: Path):
    response = upload(client, "policy.txt", "Annual leave is 20 days per year for ExampleCo staff.")
    assert response.status_code == 201
    result = client.post("/query", json={"question": "How many annual leave days?"}).json()
    assert result["citations"][0]["title"] == "policy.txt"
    assert "20 days" in result["answer"]
    assert (tmp_path / "index.json").exists()


def test_private_group_requires_assigned_reader_token(client: TestClient):
    assert upload(client, "team.txt", "Team launch date is 14 April.", "team").status_code == 201
    public = client.post("/query", json={"question": "What is the team launch date?"}).json()
    assert public["citations"] == []
    member = client.post(
        "/query",
        headers={"X-Access-Token": "test-reader-token"},
        json={"question": "What is the team launch date?"},
    ).json()
    assert member["citations"][0]["title"] == "team.txt"


def test_ingestion_requires_admin_key(client: TestClient):
    response = client.post("/documents", files={"file": ("policy.txt", b"secret", "text/plain")})
    assert response.status_code == 403
    assert client.get("/health").json()["chunks"] == 0


def test_unanswerable_question_refuses(client: TestClient):
    upload(client, "leave.txt", "Annual leave is 20 days per year.")
    result = client.post("/query", json={"question": "Who won the moon contest?"}).json()
    assert result["citations"] == []
    assert "don't know" in result["answer"]


def test_unsupported_and_oversized_documents(client: TestClient):
    assert upload(client, "file.csv", "a,b").status_code == 422
    assert upload(client, "large.txt", "x" * (2 * 1024 * 1024 + 1)).status_code == 422


def test_chunking_and_pdf_error_controls():
    chunks = chunk_text(" ".join(["alpha"] * 200), size=120, overlap=20)
    assert len(chunks) > 1
    assert all(len(chunk) <= 120 for chunk in chunks)
    with pytest.raises(ValueError):
        chunk_text("text", size=80)
    with pytest.raises(ValueError):
        extract_text("scan.pdf", b"not-a-pdf")


def test_query_validation(client: TestClient):
    assert client.post("/query", json={"question": "x"}).status_code == 422
    assert client.post("/query", json={"question": "valid", "top_k": 100}).status_code == 422
