import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, util
from cross_encoder import CrossEncoder
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
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

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
        expansion_terms = " ".join([documents[i].split()[:10] for i in top_idx])
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
    # Chunk candidates and rerank with Cross-Encoder
    # ------------------------
    rerank_inputs = []
    rerank_metadata = []
    for idx in ranked_idx[:top_k*3]:  # top candidates
        chunks = chunk_text(documents[idx])
        for chunk in chunks:
            rerank_inputs.append((query, chunk))
            rerank_metadata.append({"id": ids[idx], "title": metadatas[idx]["title"], "text": chunk})

    cross_scores = cross_encoder.predict(rerank_inputs)
    reranked = sorted(zip(rerank_metadata, cross_scores), key=lambda x: x[1], reverse=True)

    # ------------------------
    # Apply similarity threshold and return top-k
    # ------------------------
    final_results = []
    for meta, score in reranked:
        if score >= similarity_threshold:
            final_results.append({**meta, "score": float(score)})
        if len(final_results) >= top_k:
            break

    return final_results

# ========================
# Test / Evaluation
# ========================
if __name__ == "__main__":
    query = "Why did Rocky act cowardly before his execution in Angels with Dirty Faces?"
    results = retrieve(query, top_k=5)

    print("\n=== Human-Level Retrieval Results ===")
    for r in results:
        print(f"Title: {r['title']} | Score: {r['score']:.4f}")
        print(f"Chunk: {r['text'][:200]}...\n")





