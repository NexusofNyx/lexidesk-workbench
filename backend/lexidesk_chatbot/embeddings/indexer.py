"""
Embed chunks and store them in ChromaDB
(Windows-safe, robust, production-ready, debug-enhanced)
"""

import json
import uuid
from pathlib import Path
from typing import List, Dict

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# --------------------------------------------------
# Resolve paths correctly
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]  # lexidesk_chatbot/
DATA_DIR = BASE_DIR / "data"
CHUNKS_FILE = DATA_DIR / "chunks.jsonl"
CHROMA_DIR = DATA_DIR / "chroma_db"

print("\n" + "=" * 80)
print("📂 DATA_DIR    :", DATA_DIR)
print("📄 CHUNKS_FILE :", CHUNKS_FILE)
print("🧠 CHROMA_DIR  :", CHROMA_DIR)
print("=" * 80)

if not CHUNKS_FILE.exists():
    raise FileNotFoundError(f"❌ Missing chunks file: {CHUNKS_FILE}")

CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Load chunks from JSONL
# --------------------------------------------------
def load_chunks() -> List[Dict]:
    chunks: List[Dict] = []

    with CHUNKS_FILE.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    chunks.append(obj)
                else:
                    print(f"⚠️ Line {lineno}: Not a JSON object → skipped")
            except json.JSONDecodeError as e:
                print(f"⚠️ Line {lineno}: JSON decode error → {e}")

    print(f"📄 Loaded {len(chunks)} total chunk objects from disk")
    return chunks

# --------------------------------------------------
# Build Chroma index
# --------------------------------------------------
def build_index(
    model_name: str = "all-MiniLM-L6-v2",
    collection_name: str = "legal_docs",
    reset: bool = True,
):
    print("\n🚀 Starting Chroma indexing pipeline")

    # --------------------------------------------------
    # Load chunks
    # --------------------------------------------------
    chunks = load_chunks()
    if not chunks:
        raise ValueError("❌ No chunks found to index")

    # --------------------------------------------------
    # Load embedding model
    # --------------------------------------------------
    print("\n🧠 Loading embedding model:", model_name)
    model = SentenceTransformer(model_name)

    texts: List[str] = []
    metadatas: List[Dict] = []
    ids: List[str] = []

    filtered = 0

    # --------------------------------------------------
    # Prepare documents
    # --------------------------------------------------
    for chunk in chunks:
        text = (chunk.get("text") or "").strip()

        if not text or len(text.split()) < 5:
            filtered += 1
            continue

        doc_id = str(chunk.get("doc_id") or "unknown_doc")
        page = chunk.get("page")
        page = page if isinstance(page, int) else 0

        texts.append(text)
        metadatas.append({
            "doc_id": doc_id,
            "page": page,
            "source": doc_id,
        })

        # 🔐 Always unique
        ids.append(str(uuid.uuid4()))

    print(f"✅ Clean chunks ready   : {len(texts)}")
    print(f"🚮 Filtered chunks     : {filtered}")

    if not texts:
        raise ValueError("❌ All chunks were filtered out")

    # --------------------------------------------------
    # Initialize ChromaDB
    # --------------------------------------------------
    print("\n📦 Initializing ChromaDB client")

    client = chromadb.Client(
        Settings(
            persist_directory=str(CHROMA_DIR),
            anonymized_telemetry=False,
        )
    )

    # --------------------------------------------------
    # Reset collection safely
    # --------------------------------------------------
    existing = [c.name for c in client.list_collections()]
    print("📚 Existing collections:", existing)

    if reset and collection_name in existing:
        print(f"♻️ Deleting existing collection: {collection_name}")
        client.delete_collection(collection_name)

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # --------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------
    print("\n🔢 Generating embeddings...")
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    if len(embeddings) != len(texts):
        raise RuntimeError("❌ Embedding count mismatch")

    print(f"✅ Generated embeddings: {len(embeddings)}")

    # --------------------------------------------------
    # Store embeddings
    # --------------------------------------------------
    print("\n💾 Adding documents to ChromaDB...")
    try:
        collection.add(
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=ids,
        )
    except Exception as e:
        print("❌ ChromaDB insert failed")
        raise RuntimeError(e)

    # --------------------------------------------------
    # Persist DB (CRITICAL)
    # --------------------------------------------------
    print("\n💾 Persisting ChromaDB to disk...")
    try:
        client.persist()
    except Exception as e:
        print("❌ Chroma persist failed")
        raise RuntimeError(e)

    # --------------------------------------------------
    # Final verification
    # --------------------------------------------------
    count = collection.count()
    print("\n" + "=" * 80)
    print("✅ INDEXING COMPLETE")
    print(f"📊 Stored vectors : {count}")
    print(f"📁 DB location   : {CHROMA_DIR}")

    if count == 0:
        print("⚠️ WARNING: Collection is empty after indexing!")

    # Sample sanity check
    if count > 0:
        sample = collection.peek()
        print("🔍 Sample metadata:", sample["metadatas"][0])

    print("=" * 80)

# --------------------------------------------------
# CLI entry
# --------------------------------------------------
if __name__ == "__main__":
    build_index(reset=True)
