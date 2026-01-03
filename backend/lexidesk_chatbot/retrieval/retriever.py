"""
Retriever + Gemini Answer Generator
(LexiDesk – Windows-safe, stable version)
"""

from pathlib import Path
import json
import os
import uuid
from typing import List, Dict

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

import google.generativeai as genai

# --------------------------------------------------
# Load environment variables EARLY (reload-safe)
# --------------------------------------------------
BASE_BACKEND_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = BASE_BACKEND_DIR / ".env"

if ENV_PATH.exists():
    from dotenv import load_dotenv
    load_dotenv(ENV_PATH)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("❌ GEMINI_API_KEY not found in .env")

genai.configure(api_key=GEMINI_API_KEY)

# --------------------------------------------------
# Paths
# --------------------------------------------------
DATA_DIR = BASE_BACKEND_DIR / "lexidesk_chatbot" / "data"
CHUNKS_FILE = DATA_DIR / "chunks.jsonl"
CHROMA_DIR = DATA_DIR / "chroma_db"

DATA_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Embedding model (LOAD ONCE)
# --------------------------------------------------
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBED_MODEL_NAME)

# --------------------------------------------------
# Chroma client (GLOBAL, SAFE)
# --------------------------------------------------
client = chromadb.Client(
    Settings(
        persist_directory=str(CHROMA_DIR),
        anonymized_telemetry=False,
    )
)

COLLECTION_NAME = "legal_docs"

def get_collection():
    """Always return a valid Chroma collection."""
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

# --------------------------------------------------
# Index chunks into Chroma
# --------------------------------------------------
def index_chunks(force: bool = False):
    """
    Index chunks.jsonl into ChromaDB.
    Safe to call multiple times.
    """
    if force:
        try:
            client.delete_collection(COLLECTION_NAME)
            print("🗑️ Existing Chroma collection deleted.")
        except Exception:
            pass

    collection = get_collection()

    if collection.count() > 0 and not force:
        print("ℹ️ Chroma collection already indexed.")
        return

    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"❌ Missing chunks file: {CHUNKS_FILE}")

    documents: List[str] = []
    metadatas: List[Dict] = []
    ids: List[str] = []

    with CHUNKS_FILE.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                print(f"⚠️ Skipping malformed JSON at line {lineno}")
                continue

            text = (chunk.get("text") or "").strip()
            if not text or len(text.split()) < 5:
                continue

            doc_id = str(chunk.get("doc_id") or "unknown_doc")
            page = chunk.get("page")
            page = page if isinstance(page, int) else -1

            documents.append(text)
            metadatas.append({
                "doc_id": doc_id,
                "page": page,
                "source": doc_id,
            })

            # 🔐 Always unique → prevents 500 crashes
            ids.append(str(uuid.uuid4()))

    if not documents:
        raise RuntimeError("❌ No valid text chunks found to index.")

    print(f"🔢 Embedding {len(documents)} chunks...")

    embeddings = embedder.encode(
        documents,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    collection.add(
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
        ids=ids,
    )

    client.persist()
    print("✅ Chroma indexing complete.")

# --------------------------------------------------
# Retrieve top-k chunks
# --------------------------------------------------
def retrieve(query: str, top_k: int = 5) -> List[Dict]:
    collection = get_collection()

    if collection.count() == 0:
        print("⚠️ Chroma collection is empty.")
        return []

    query_embedding = embedder.encode(
        [query],
        normalize_embeddings=True,
    )

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k,
    )

    retrieved = []

    if results.get("documents"):
        for i in range(len(results["documents"][0])):
            retrieved.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": results["distances"][0][i],
            })

    if not retrieved:
        print("⚠️ No relevant chunks found.")
    else:
        print(f"ℹ️ Retrieved {len(retrieved)} chunks:")
        for r in retrieved:
            print(
                f"- Page {r['metadata']['page']} | "
                f"{r['metadata']['doc_id']} | "
                f"distance={r['score']:.4f}"
            )

    return retrieved

# --------------------------------------------------
# Gemini answer generation
# --------------------------------------------------
def generate_answer(query: str, retrieved_chunks: List[Dict]) -> str:
    if not retrieved_chunks:
        return "The provided documents do not contain this information."

    context = "\n\n".join(
        f"(Page {c['metadata']['page']}) {c['text']}"
        for c in retrieved_chunks
    )

    prompt = f"""
You are a legal assistant.

Answer using ONLY the information present in the context below.
If the answer is not present, say exactly:
"The provided documents do not contain this information."

Context:
{context}

Question:
{query}

Answer:
""".strip()

    model = genai.GenerativeModel("gemini-1.5-flash")

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ Gemini generation error: {e}"

# --------------------------------------------------
# CLI testing
# --------------------------------------------------
if __name__ == "__main__":
    index_chunks(force=True)

    while True:
        query = input("\nAsk a legal question (or 'exit'): ").strip()
        if query.lower() == "exit":
            break

        retrieved = retrieve(query)
        answer = generate_answer(query, retrieved)

        print("\n" + "=" * 80)
        print("ANSWER:\n")
        print(answer)
        print("\nSOURCES:\n")
        for r in retrieved:
            print(f"- Page {r['metadata']['page']} | {r['metadata']['doc_id']}")
