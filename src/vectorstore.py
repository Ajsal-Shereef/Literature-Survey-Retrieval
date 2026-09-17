import pickle
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from ingest import chunk_documents, load_documents

INDEX_DIR = Path(__file__).parent.parent / "index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_model():
    return SentenceTransformer(EMBEDDING_MODEL)


def build(chunk_size=500, chunk_overlap=50):
    docs = load_documents()
    chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    texts = [c["text"] for c in chunks]
    embeddings = get_model().encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    INDEX_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(INDEX_DIR / "index.faiss"))
    with open(INDEX_DIR / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print(f"Built index: {len(chunks)} vectors, dim={dim}")
    return index, chunks


def load():
    index = faiss.read_index(str(INDEX_DIR / "index.faiss"))
    with open(INDEX_DIR / "chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


def search(query, k=5):
    index, chunks = load()
    query_vec = get_model().encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = index.search(query_vec, k)
    return [{**chunks[i], "score": float(s)} for s, i in zip(scores[0], indices[0])]


if __name__ == "__main__":
    build()
    for r in search("What is policy reuse in reinforcement learning?"):
        print(f"[{r['score']:.3f}] {r['metadata']['file']} (chunk {r['metadata']['chunk_index']})")
        print(r["text"][:150].replace("\n", " "))
        print()
