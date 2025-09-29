import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
from groq import Groq
import os
from dotenv import load_dotenv
import json
import re

# ==============================
# Load API Key
# ==============================
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables")
client = Groq(api_key=api_key)

# ==============================
# File paths
# ==============================
INDEX_FILE = "faiss_movie_plots.index"
METADATA_FILE = "faiss_metadata.pkl"

# ==============================
# Load FAISS index and metadata
# ==============================
index = faiss.read_index(INDEX_FILE)
with open(METADATA_FILE, "rb") as f:
    metadata = pickle.load(f)

ids = metadata["ids"]
documents = metadata["documents"]
metadatas = metadata["metadatas"]

# ==============================
# Load embedding model
# ==============================
embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# ==============================
# Preprocess for BM25 (optional sparse retrieval)
# ==============================
tokenized_docs = [doc.split() for doc in documents]
bm25 = BM25Okapi(tokenized_docs)

# ==============================
# Helper: Chunking with overlap
# ==============================
def chunk_text(text, chunk_size=150, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks

# ==============================
# Retrieval function
# ==============================
def retrieve_contexts(query, top_k=5, similarity_threshold=0.5, use_bm25=True, bm25_weight=0.3):
    # Optional BM25 expansion
    if use_bm25:
        bm25_scores = bm25.get_scores(query.split())
    else:
        bm25_scores = np.zeros(len(documents))
    
    # Dense embedding
    query_vec = embed_model.encode([query], normalize_embeddings=True).astype(np.float32)
    distances, indices = index.search(query_vec, top_k*3)  # get more candidates for reranking

    candidates = []
    for idx in indices[0]:
        if 0 <= idx < len(documents):
            dense_score = 1 - distances[0][list(indices[0]).index(idx)]  # convert FAISS distance to similarity
            combined_score = (1-bm25_weight)*dense_score + bm25_weight*bm25_scores[idx]
            candidates.append((idx, combined_score))

    # Sort by combined score
    candidates.sort(key=lambda x: x[1], reverse=True)

    # Chunk candidates and compute cosine similarity
    query_embedding = embed_model.encode(query, normalize_embeddings=True)
    rerank_results = []
    for idx, _ in candidates[:top_k*3]:
        chunks = chunk_text(documents[idx])
        for chunk in chunks:
            chunk_embedding = embed_model.encode(chunk, normalize_embeddings=True)
            sim_score = util.cos_sim(query_embedding, chunk_embedding).item()
            if sim_score >= similarity_threshold:
                rerank_results.append({
                    "id": ids[idx],
                    "title": metadatas[idx]["title"],
                    "text": chunk,
                    "score": sim_score
                })

    # Sort by similarity score and return top-k
    rerank_results.sort(key=lambda x: x['score'], reverse=True)
    return rerank_results[:top_k]

# ==============================
# Generate answer with Groq LLM
# ==============================
def generate_answer(query, contexts):
    numbered_contexts = [f"[{i+1}] {ctx}" for i, ctx in enumerate(contexts)]
    context_text = "\n\n".join(numbered_contexts)

    system_prompt = """You are a movie plot expert that answers questions using ONLY the provided plot snippets.
You must respond with valid JSON containing three fields: answer, contexts, and reasoning.
Always use information from the provided snippets to answer the question."""

    user_prompt = f"""Question: {query}

Movie Plot Snippets:
{context_text}

Instructions:
1. Read ALL the plot snippets carefully
2. Find which snippet(s) answer the question
3. Provide a clear, natural language answer based on those snippets
4. Include the relevant snippet text in the "contexts" array
5. Explain your reasoning

Output format (JSON only, no markdown):
{{
  "answer": "Your natural language answer here",
  "contexts": ["Relevant snippet 1", "Relevant snippet 2"],
  "reasoning": "Brief explanation of how you found the answer"
}}"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=800
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling API: {e}")
        return json.dumps({
            "answer": "Error generating answer",
            "contexts": [],
            "reasoning": f"API Error: {str(e)}"
        })

# ==============================
# Safe JSON parser
# ==============================
def parse_json_safe(output_text):
    if not output_text or not isinstance(output_text, str):
        return None
    try:
        return json.loads(output_text)
    except json.JSONDecodeError:
        pass
    # Clean up and attempt parsing
    cleaned = re.sub(r"```json|```", "", output_text).strip()
    try:
        return json.loads(cleaned)
    except:
        pass
    # Extract JSON between braces
    brace_count = 0
    start_idx = -1
    for i, c in enumerate(cleaned):
        if c == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif c == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                try:
                    return json.loads(cleaned[start_idx:i+1])
                except:
                    pass
    return None

# ==============================
# Main
# ==============================
if __name__ == "__main__":
    user_query = "Angels with Dirty Faces, why did Rocky ultimately act cowardly before his execution, and what was the real purpose behind this behavior?"

    print("\n=== MOVIE PLOT RAG SYSTEM ===\n")
    print(f"Query: {user_query}\n")

    # Retrieve contexts
    print("Retrieving relevant contexts...")
    contexts = retrieve_contexts(user_query, top_k=5)
    print(f"Found {len(contexts)} contexts\n")

    # Show contexts
    for i, ctx in enumerate(contexts, 1):
        preview = ctx['text'][:150] + "..." if len(ctx['text']) > 150 else ctx['text']
        print(f"[{i}] Title: {ctx['title']} | Score: {ctx['score']:.4f}\nPreview: {preview}\n")

    # Generate answer
    print("Generating answer from LLM...")
    output_json = generate_answer(user_query, [c['text'] for c in contexts])

    result = parse_json_safe(output_json)
    print("\n=== RESULT ===\n")
    if result:
        print(json.dumps(result, indent=2))
    else:
        print("❌ Could not parse JSON response")
        print("Raw output:", output_json)



