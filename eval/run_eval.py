"""Small reproducible retrieval/citation check over the synthetic sample corpus."""

import json
import os
import sys
from pathlib import Path
from time import perf_counter

from fastapi.testclient import TestClient

from rag_assistant.api import create_app
from rag_assistant.config import Settings


def main() -> None:
    os.environ.setdefault("RAG_EMBEDDING_BACKEND", "hash")
    os.environ.setdefault("RAG_GENERATOR", "extractive")
    settings = Settings.from_env()
    if not settings.index_path.exists():
        sys.exit("Index missing. Run: python -m rag_assistant ingest samples --reset")
    client = TestClient(create_app(settings))
    cases = json.loads(Path("eval/questions.json").read_text(encoding="utf-8"))
    correct = 0
    latencies = []
    failures = []
    for case in cases:
        start = perf_counter()
        response = client.post("/query", json={"question": case["question"]})
        response.raise_for_status()
        result = response.json()
        latencies.append(round((perf_counter() - start) * 1000, 2))
        titles = {source["title"] for source in result["citations"]}
        passed = (case["source"] in titles) if case["source"] else not titles
        correct += int(passed)
        if not passed:
            failures.append(
                {"question": case["question"], "expected": case["source"], "seen": sorted(titles)}
            )
    report = {
        "cases": len(cases),
        "citation_or_refusal_passes": correct,
        "latency_ms_per_case": latencies,
        "cost_usd": None,
        "cost_note": "Extractive mode uses no LLM API; compute provider cost from usage and current provider pricing.",
        "failures": failures,
    }
    print(json.dumps(report, indent=2))
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
