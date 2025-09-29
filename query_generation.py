import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
import os
from dotenv import load_dotenv
import json
from difflib import SequenceMatcher


load_dotenv()
print("GROQ_API_KEY loaded:", os.getenv("GROQ_API_KEY"))
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


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


def retrieve_contexts(query, top_k=3):
    """
    Embed query, search FAISS, return top-k contexts
    """
    query_vec = embed_model.encode([query], normalize_embeddings=True)
    query_vec = np.array(query_vec, dtype=np.float32)
    distances, indices = index.search(query_vec, top_k)

    contexts = []
    for i in indices[0]:
        if 0 <= i < len(documents):
            contexts.append(documents[i])
    return contexts


def generate_answer(query, contexts):
    """
    Calls Groq API with retrieved contexts to generate structured JSON
    """
    context_text = "\n".join(contexts)

    prompt = f"""
You are a helpful assistant. Use the given movie plot snippets to answer the user’s question.
Return output ONLY in valid JSON with keys: answer, contexts, reasoning.

Question: {query}

Relevant Plot Snippets:
{context_text}
    """

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=400
    )

    return response.choices[0].message.content


def evaluate_output(output_json, ground_truth_answer, ground_truth_contexts):
    """
    Evaluate model output against ground truth.
    Returns a dictionary of metrics.
    """
    try:
        result = json.loads(output_json)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON returned from model"}

    predicted_answer = result.get("answer", "")
    predicted_contexts = result.get("contexts", [])

    # --- 1. Answer Accuracy (string similarity) ---
    answer_similarity = SequenceMatcher(None, predicted_answer.lower(), ground_truth_answer.lower()).ratio()

    # --- 2. Context Precision & Recall ---
    relevant = 0
    for ctx in predicted_contexts:
        if any(gt.lower() in ctx.lower() for gt in ground_truth_contexts):
            relevant += 1

    precision = relevant / len(predicted_contexts) if predicted_contexts else 0
    recall = relevant / len(ground_truth_contexts) if ground_truth_contexts else 0

    return {
        "answer_similarity": round(answer_similarity, 3),
        "context_precision": round(precision, 3),
        "context_recall": round(recall, 3),
        "predicted_answer": predicted_answer,
        "predicted_contexts": predicted_contexts
    }


if __name__ == "__main__":
    # Example query for Zardoz
    user_query = "Which movie features an artificial intelligence called the Tabernacle?"
    
    ground_truth_answer = (
        "The movie *Zardoz* features an artificial intelligence system called the Tabernacle, "
        "which oversees and protects the Eternals from death."
    )
    
    ground_truth_contexts = [
        "Zardoz ... The Eternals are overseen and protected from death by the Tabernacle, an artificial intelligence.",
        "The Vortex developed complex social rules whose violators are punished ... by the Tabernacle."
    ]

    # Retrieval + Generation
    contexts = retrieve_contexts(user_query, top_k=3)
    output_json = generate_answer(user_query, contexts)
    print("Model Output:", output_json)

    # Evaluation
    metrics = evaluate_output(output_json, ground_truth_answer, ground_truth_contexts)
    print("Evaluation Metrics:", metrics)
