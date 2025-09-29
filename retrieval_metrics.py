import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi

# ========================
# Config / File Paths
# ========================
INDEX_FILE = "faiss_movie_plots.index"
METADATA_FILE = "faiss_metadata.pkl"

# ========================
# Load FAISS index & metadata
# ========================
index = faiss.read_index(INDEX_FILE)
with open(METADATA_FILE, "rb") as f:
    metadata = pickle.load(f)

ids = metadata["ids"]
documents = metadata["documents"]
metadatas = metadata["metadatas"]

# ========================
# Load models
# ========================
dense_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ========================
# Preprocess for BM25 (sparse retrieval)
# ========================
tokenized_docs = [doc.split() for doc in documents]
bm25 = BM25Okapi(tokenized_docs)

# ========================
# Helper: chunking with overlap
# ========================
def chunk_text(text, chunk_size=150, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks

# ========================
# Retrieval function
# ========================
def retrieve(query, top_k=5, bm25_weight=0.3, similarity_threshold=0.5, use_query_expansion=True):
    # ------------------------
    # Optional semantic query expansion
    # ------------------------
    if use_query_expansion:
        # Get nearest neighbors in embedding space
        query_vec = dense_model.encode([query], normalize_embeddings=True)
        all_doc_vecs = dense_model.encode(documents, normalize_embeddings=True)
        sims = util.cos_sim(query_vec, all_doc_vecs)[0].cpu().numpy()
        top_idx = sims.argsort()[-3:]  # pick top-3 semantically similar docs
        expansion_terms = " ".join([" ".join(documents[i].split()[:10]) for i in top_idx])
        query += " " + expansion_terms

    # ------------------------
    # Dense retrieval (FAISS)
    # ------------------------
    query_vec = dense_model.encode([query], normalize_embeddings=True).astype(np.float32)
    distances, indices = index.search(query_vec, top_k*3)  # get more candidates for reranking

    # ------------------------
    # Sparse retrieval (BM25)
    # ------------------------
    bm25_scores = bm25.get_scores(query.split())
    
    # Combine dense + sparse
    combined_scores = []
    candidates = []
    for idx in indices[0]:
        score = (1-bm25_weight) * (1 - distances[0][list(indices[0]).index(idx)]) + bm25_weight * bm25_scores[idx]
        combined_scores.append(score)
        candidates.append(idx)

    # Rank by combined score
    ranked_idx = [x for _, x in sorted(zip(combined_scores, candidates), reverse=True)]

    # ------------------------
    # Chunk candidates and rerank with SentenceTransformer cosine similarity
    # ------------------------
    # Encode query once
    query_embedding = dense_model.encode(query, normalize_embeddings=True)
    
    rerank_results = []
    for idx in ranked_idx[:top_k*3]:  # top candidates
        chunks = chunk_text(documents[idx])
        for chunk in chunks:
            # Encode chunk and compute cosine similarity
            chunk_embedding = dense_model.encode(chunk, normalize_embeddings=True)
            similarity_score = util.cos_sim(query_embedding, chunk_embedding).item()
            
            rerank_results.append({
                "id": ids[idx],
                "title": metadatas[idx]["title"],
                "text": chunk,
                "score": similarity_score
            })
    
    # Sort by similarity score
    rerank_results.sort(key=lambda x: x['score'], reverse=True)

    # ------------------------
    # Apply similarity threshold and return top-k
    # ------------------------
    final_results = []
    for result in rerank_results:
        if result['score'] >= similarity_threshold:
            final_results.append(result)
        if len(final_results) >= top_k:
            break

    return final_results

# ========================
# Test / Evaluation
# ========================
def evaluate_retrieval(query, ground_truth_text, top_k=5, similarity_threshold=0.5):
    """
    Evaluate retrieval metrics for human-level pipeline
    """
    retrieved = retrieve(query, top_k=top_k, similarity_threshold=similarity_threshold)
    gt_vec = dense_model.encode([ground_truth_text], normalize_embeddings=True)

    relevances = []
    scores = []

    for r in retrieved:
        chunk_vec = dense_model.encode(r['text'], normalize_embeddings=True)
        sim = util.cos_sim(gt_vec, chunk_vec).item()
        scores.append({"chunk": r['text'][:200], "similarity": sim, "title": r['title']})
        relevances.append(1 if sim >= similarity_threshold else 0)

    # --- Metrics ---
    precision = sum(relevances)/top_k if top_k else 0
    recall = sum(relevances)/1  # assuming single ground truth
    f1 = (2*precision*recall)/(precision+recall) if (precision+recall) else 0

    # MRR
    mrr = 0
    for rank, rel in enumerate(relevances, start=1):
        if rel == 1:
            mrr = 1/rank
            break

    # Hit Rate
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
        "nDCG@k": ndcg
    }

    return scores, metrics



if __name__ == "__main__":
    query = "Why did Rocky act cowardly before his execution in Angels with Dirty Faces?"
    ground_truth = """Rocky pretended to be a coward by begging and screaming for mercy on his way to the electric chair. He did this at Jerry's request to destroy his heroic image in the eyes of the gang of boys (Soapy and friends) who idolized him, so they would lose respect for criminal behavior and have a chance at better lives."""

    retrieved_scores, metrics = evaluate_retrieval(query, ground_truth, top_k=5)

    print("\n=== Retrieved Chunks ===")
    for r in retrieved_scores:
        print(f"Title: {r['title']} | Similarity: {r['similarity']:.4f}")
        print(f"Chunk: {r['chunk']}...\n")

    print("=== Retrieval Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")






