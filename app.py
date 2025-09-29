import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# File paths
INDEX_FILE = "faiss_movie_plots.index"
METADATA_FILE = "faiss_metadata.pkl"

# Load index & metadata
index = faiss.read_index(INDEX_FILE)
with open(METADATA_FILE, "rb") as f:
    metadata = pickle.load(f)

ids = metadata["ids"]
documents = metadata["documents"]
metadatas = metadata["metadatas"]

# Load the same model you used for embedding
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def search(query, top_k=5):
    # Encode query
    query_vec = model.encode([query], normalize_embeddings=True)
    query_vec = np.array(query_vec, dtype=np.float32)

    # Search FAISS index
    distances, indices = index.search(query_vec, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        result = {
            "id": ids[idx],
            "text": documents[idx],
            "metadata": metadatas[idx],
            "distance": float(dist),
        }
        results.append(result)

    return results


if __name__ == "__main__":
    query = "A young wizard attending a magical school"
    results = search(query, top_k=3)

    print("\nSearch Results:")
    for r in results:
        print(f"Title: {r['metadata']['title']} | Distance: {r['distance']:.4f}")
        print(f"Chunk: {r['text'][:200]}...\n")

