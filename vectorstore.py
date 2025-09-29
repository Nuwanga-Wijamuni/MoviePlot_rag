import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os
from dotenv import load_dotenv
import logging
import pickle

# Set up logging
logging.basicConfig(level=logging.WARNING)

def chunk_text(text, chunk_size=300):
    """
    Splits a text into chunks of a specified number of words.
    """
    if not isinstance(text, str):
        return []
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


def process_and_embed_data(csv_path, index_file, metadata_file):
    """
    Loads data from a CSV, chunks the plots, embeds them, and stores them in FAISS.
    """
    load_dotenv()
    print(f"Loading movie plots from '{csv_path}'...")

    try:
        df = pd.read_csv(csv_path)
        df.dropna(subset=['Plot'], inplace=True)

        # Limit the rows (optional)
        df = df.head(500)
        print(f"Successfully loaded and reduced to {len(df)} movies.")

    except FileNotFoundError:
        print(f"Error: The file '{csv_path}' was not found.")
        return

    # --- Initialize Sentence Transformer model ---
    print("Initializing Sentence Transformer model...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    dim = model.get_sentence_embedding_dimension()
    print(f"Embedding dimension: {dim}")

    # --- Prepare documents ---
    print("Preparing documents, metadata, and IDs...")
    documents = []
    metadatas = []
    ids = []

    for index, row in df.iterrows():
        title = row['Title']
        plot = row['Plot']
        plot_chunks = chunk_text(plot)
        for i, chunk in enumerate(plot_chunks):
            documents.append(chunk)
            metadatas.append({'title': str(title), 'chunk_index': i})
            ids.append(f"id_{index}_{i}")

    if not documents:
        print("No documents to process. Exiting.")
        return

    # --- Generate embeddings ---
    print("Generating embeddings...")
    embeddings = model.encode(documents, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype=np.float32)  # FAISS needs float32
    print(f"Generated {len(embeddings)} embeddings.")

    # --- Create FAISS index ---
    print("Creating FAISS index...")
    index = faiss.IndexFlatL2(dim)  # L2 distance
    index.add(embeddings)
    print(f"FAISS index contains {index.ntotal} vectors.")

    # --- Save FAISS index and metadata ---
    print("Saving FAISS index and metadata...")
    faiss.write_index(index, index_file)
    with open(metadata_file, "wb") as f:
        pickle.dump({"ids": ids, "documents": documents, "metadatas": metadatas}, f)

    print("\n--- Processing Complete ---")
    print(f"Index saved to: {index_file}")
    print(f"Metadata saved to: {metadata_file}")


if __name__ == '__main__':
    CSV_FILE_PATH = 'subset_movie_plots.csv'
    INDEX_FILE = 'faiss_movie_plots.index'
    METADATA_FILE = 'faiss_metadata.pkl'

    process_and_embed_data(CSV_FILE_PATH, INDEX_FILE, METADATA_FILE)