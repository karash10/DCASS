import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# -------------------------------------------------
# Paths
# -------------------------------------------------
CHUNKS_PATH = "C:\\Users\\kappa\\OneDrive\\capstone\\dcass\\data\\chunks.json"
EMBEDDINGS_PATH = "C:\\Users\\kappa\\OneDrive\\capstone\\dcass\\data\\embeddings.npy"
FAISS_INDEX_PATH = "C:\\Users\\kappa\\OneDrive\\capstone\\dcass\\data\\faiss.index"


# -------------------------------------------------
# Load chunks
# -------------------------------------------------
def load_chunks(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -------------------------------------------------
# Generate embeddings
# -------------------------------------------------
def embed_chunks(chunks, model):
    texts = [c["text"] for c in chunks]

    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    return np.array(embeddings, dtype="float32")


# -------------------------------------------------
# Build FAISS index (cosine similarity)
# -------------------------------------------------
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


# -------------------------------------------------
# Test semantic search
# -------------------------------------------------
def test_search(index, model, chunks, query, k=3):
    print(f"\n🔍 Query: {query}")

    q_emb = model.encode(
        [query],
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(q_emb, k)

    for rank, idx in enumerate(indices[0]):
        print(f"\nRank {rank + 1}")
        print(f"Score: {scores[0][rank]:.4f}")
        print(chunks[idx]["text"])


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    print("📥 Loading chunks...")
    chunks = load_chunks(CHUNKS_PATH)

    print("🧠 Loading embedding model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("🔢 Generating embeddings...")
    embeddings = embed_chunks(chunks, model)

    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"💾 Saved embeddings to {EMBEDDINGS_PATH}")

    print("📦 Building FAISS index...")
    index = build_faiss_index(embeddings)

    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"💾 Saved FAISS index to {FAISS_INDEX_PATH}")

    # -------------------------------------------------
    # Manual tests
    # -------------------------------------------------
    test_queries = [
        "lexical analysis",
        "syntax parsing",
        "compiler optimization",
        "context free grammar"
    ]

    for q in test_queries:
        test_search(index, model, chunks, q, k=3)


if __name__ == "__main__":
    main()
