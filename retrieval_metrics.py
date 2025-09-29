import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, util

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

# Load embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def search(query, top_k=5):
    """Retrieve top-k relevant chunks"""
    query_vec = model.encode([query], normalize_embeddings=True)
    query_vec = np.array(query_vec, dtype=np.float32)

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


def evaluate_accuracy(query, ground_truth_text, top_k=5, threshold=0.6):
    """
    Evaluate retrieval quality with multiple IR metrics.
    threshold = cosine similarity cutoff to treat as 'relevant'
    """
    retrieved = search(query, top_k=top_k)
    gt_vec = model.encode([ground_truth_text], normalize_embeddings=True)

    # Track relevance
    relevances = []
    scores = []

    for r in retrieved:
        chunk_vec = model.encode([r["text"]], normalize_embeddings=True)
        sim = util.cos_sim(gt_vec, chunk_vec).item()
        scores.append({"chunk": r["text"][:200], "similarity": sim, "title": r["metadata"]["title"]})
        relevances.append(1 if sim >= threshold else 0)

    # Metrics
    precision = sum(relevances) / top_k if top_k > 0 else 0
    recall = sum(relevances) / 1  # assuming 1 ground truth text
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0

    # MRR (first relevant result rank)
    mrr = 0
    for rank, rel in enumerate(relevances, start=1):
        if rel == 1:
            mrr = 1 / rank
            break

    # Hit Rate (any relevant found)
    hit_rate = 1 if sum(relevances) > 0 else 0

    # nDCG
    dcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevances))
    ideal_dcg = sum(1 / np.log2(idx + 2) for idx in range(sum(relevances)))
    ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0

    metrics = {
        "Precision@k": precision,
        "Recall@k": recall,
        "F1": f1,
        "MRR": mrr,
        "HitRate@k": hit_rate,
        "nDCG@k": ndcg,
    }

    return scores, metrics


if __name__ == "__main__":
    query = "A humanoid alien arrives in Washington D.C. with a robot"
    ground_truth = """When a flying saucer lands in Washington, D.C., a humanoid alien named Klaatu arrives with a robot named Gort."""

    results, metrics = evaluate_accuracy(query, ground_truth, top_k=5)

    print("\nRetrieved Chunks:")
    for r in results:
        print(f"Title: {r['title']} | Similarity: {r['similarity']:.4f}")
        print(f"Chunk: {r['chunk']}...\n")

    print("Evaluation Metrics:")
    for m, v in metrics.items():
        print(f"{m}: {v:.4f}")



