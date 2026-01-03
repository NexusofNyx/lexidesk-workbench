"""
Ingest PDFs -> page-level text -> sentence segmentation (CNN + CRF)
Produces:
- lexidesk_chatbot/data/lexidesk_pages.jsonl
- lexidesk_chatbot/data/chunks.jsonl
"""

from pathlib import Path
import json
import fitz  # PyMuPDF
import re
from tempfile import NamedTemporaryFile
from typing import List

# --------------------------------------------------
# Correct import (NO indexer here)
# --------------------------------------------------
from lexidesk_chatbot.segmenter import segment_text

# --------------------------------------------------
# Resolve backend root
# --------------------------------------------------
BACKEND_ROOT = Path(__file__).resolve().parents[2]

# --------------------------------------------------
# Data directory (SINGLE SOURCE OF TRUTH)
# --------------------------------------------------
DATA_DIR = BACKEND_ROOT / "lexidesk_chatbot" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

PAGES_JSONL = DATA_DIR / "lexidesk_pages.jsonl"
CHUNKS_JSONL = DATA_DIR / "chunks.jsonl"

# --------------------------------------------------
# PDF extraction
# --------------------------------------------------
def extract_pages_from_pdf(pdf_path: str) -> List[dict]:
    doc = fitz.open(pdf_path)
    pages = []

    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if not text:
            continue

        pages.append({
            "doc_id": Path(pdf_path).name,
            "page": i + 1,
            "text": text,
        })

    doc.close()
    return pages


# --------------------------------------------------
# Sentence segmentation
# --------------------------------------------------
def segment_pages(pages: List[dict]) -> List[dict]:
    segmented = []

    for p in pages:
        try:
            sentences = segment_text(p["text"])
            if isinstance(sentences, tuple):
                sentences = sentences[0]

        except Exception as e:
            print(f"[WARN] Segmentation failed (page {p['page']}): {e}")
            sentences = [
                s.strip()
                for s in re.split(r"(?<=[.!?])\s+", p["text"])
                if s.strip()
            ]

        if not sentences:
            continue

        segmented.append({
            "doc_id": p["doc_id"],
            "page": p["page"],
            "sentences": sentences,
        })

    return segmented


# --------------------------------------------------
# Chunking for RAG
# --------------------------------------------------
def chunk_pages(segmented_pages: List[dict], max_chars: int = 1600) -> List[dict]:
    chunks = []

    for p in segmented_pages:
        buffer = []
        length = 0

        for sentence in p["sentences"]:
            if length + len(sentence) > max_chars and buffer:
                chunks.append({
                    "doc_id": p["doc_id"],
                    "page": p["page"],
                    "text": " ".join(buffer),
                })
                buffer = [sentence]
                length = len(sentence)
            else:
                buffer.append(sentence)
                length += len(sentence)

        if buffer:
            chunks.append({
                "doc_id": p["doc_id"],
                "page": p["page"],
                "text": " ".join(buffer),
            })

    return chunks


# --------------------------------------------------
# Main ingestion logic
# --------------------------------------------------
def run(pdf_paths: List[str]):
    all_pages = []

    for pdf in pdf_paths:
        all_pages.extend(extract_pages_from_pdf(pdf))

    if not all_pages:
        raise ValueError("No extractable text found in PDFs")

    # Append pages
    with open(PAGES_JSONL, "a", encoding="utf-8") as f:
        for page in all_pages:
            f.write(json.dumps(page, ensure_ascii=False) + "\n")

    print(f"[Ingest] Saved {len(all_pages)} pages -> {PAGES_JSONL}")

    segmented_pages = segment_pages(all_pages)
    chunks = chunk_pages(segmented_pages)

    if not chunks:
        raise ValueError("No chunks produced after segmentation")

    # Append chunks
    with open(CHUNKS_JSONL, "a", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"[Ingest] Saved {len(chunks)} chunks -> {CHUNKS_JSONL}")

    return PAGES_JSONL, CHUNKS_JSONL


# --------------------------------------------------
# FastAPI-compatible upload ingestion
# --------------------------------------------------
def ingest_document(upload_file) -> str:
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(upload_file.file.read())
        tmp_path = tmp.name

    run([tmp_path])
    return Path(upload_file.filename).stem
