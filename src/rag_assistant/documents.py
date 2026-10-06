"""Text/PDF extraction and stable, bounded chunking."""

import re
from io import BytesIO

from pypdf import PdfReader
from pypdf.errors import PdfReadError

MAX_UPLOAD_BYTES = 2 * 1024 * 1024
MAX_PAGES = 40


def extract_text(filename: str, content: bytes) -> str:
    if not content or len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("Document must be 1 byte to 2 MiB")
    suffix = filename.lower().rsplit(".", 1)[-1]
    if suffix in {"txt", "md"}:
        result = content.decode("utf-8")
    elif suffix == "pdf":
        try:
            reader = PdfReader(BytesIO(content))
        except PdfReadError as exc:
            raise ValueError("Invalid PDF") from exc
        if reader.is_encrypted or len(reader.pages) > MAX_PAGES:
            raise ValueError("Encrypted PDFs or PDFs over 40 pages are unsupported")
        try:
            result = "\n".join(page.extract_text() or "" for page in reader.pages)
        except PdfReadError as exc:
            raise ValueError("PDF text extraction failed") from exc
    else:
        raise ValueError("Only UTF-8 .txt, .md, and text-based .pdf are supported")
    result = re.sub(r"\s+", " ", result).strip()
    if not result:
        raise ValueError("No extractable text found; scanned PDFs need OCR")
    return result


def chunk_text(text: str, size: int = 600, overlap: int = 80) -> list[str]:
    if not 100 <= size <= 2000 or not 0 <= overlap < size:
        raise ValueError("Chunk size must be 100-2000 and overlap smaller than size")
    words = text.split()
    chunks: list[str] = []
    current: list[str] = []
    for word in words:
        if current and len(" ".join(current + [word])) > size:
            chunks.append(" ".join(current))
            tail: list[str] = []
            for prior in reversed(current):
                if len(" ".join([prior] + tail)) > overlap:
                    break
                tail.insert(0, prior)
            current = tail + [word]
        else:
            current.append(word)
    if current:
        chunks.append(" ".join(current))
    return chunks
