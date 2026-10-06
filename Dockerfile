FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 RAG_INDEX_PATH=/app/state/index.json
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir '.[semantic]'
COPY samples ./samples
RUN mkdir /app/state && python -m rag_assistant ingest samples --reset
EXPOSE 8000
CMD ["uvicorn", "rag_assistant.api:app", "--host", "0.0.0.0", "--port", "8000"]
