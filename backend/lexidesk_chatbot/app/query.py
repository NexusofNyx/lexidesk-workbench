# lexidesk_chatbot/app/query.py

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from pathlib import Path

# --------------------------------------------------
# Correct BASE path (MATCH ingestion)
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]  # backend

DATA_DIR = BASE_DIR / "lexidesk_chatbot" / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"

if not CHROMA_DIR.exists():
    raise RuntimeError(f"ChromaDB directory not found: {CHROMA_DIR}")

# --------------------------------------------------
# ChromaDB setup
# --------------------------------------------------
client = chromadb.Client(
    Settings(
        persist_directory=str(CHROMA_DIR),
        anonymized_telemetry=False,
    )
)

collection = client.get_or_create_collection(
    name="legal_docs",
    metadata={"hnsw:space": "cosine"},
)

print("DEBUG: collection count =", collection.count())
print("DEBUG: collection peek =", collection.peek())

# --------------------------------------------------
# Embedding model
# --------------------------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# --------------------------------------------------
# Interactive query loop
# --------------------------------------------------
def main():
    print("\n🔎 LexiDesk Legal Search (RAG)")
    print("Type 'exit' to quit")

    while True:
        query = input("\nAsk a legal question: ").strip()
        if not query:
            continue
        if query.lower() == "exit":
            break

        q_emb = model.encode(query, normalize_embeddings=True).tolist()

        results = collection.query(
            query_embeddings=[q_emb],
            n_results=5,
        )

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        if not docs:
            print("\n❌ No relevant passages found.\n")
            continue

        print("\n📄 Top relevant passages:\n")

        for i, text in enumerate(docs):
            meta = metas[i] if metas else {}
            page = meta.get("page", "?")
            doc_id = meta.get("doc_id", "unknown")
            dist = dists[i] if dists else None

            print(f"--- Result {i+1} ---")
            print(f"Source: {doc_id} | Page: {page} | distance: {dist:.4f}")
            print(text[:700])
            print()


if __name__ == "__main__":
    main()
