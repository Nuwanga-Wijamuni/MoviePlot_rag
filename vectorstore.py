import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import os
from dotenv import load_dotenv
import chromadb 
import logging


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

def process_and_embed_data(csv_path, collection_name):
    """
    Loads data from a CSV, chunks the plots, embeds them, and stores them in ChromaDB.
    """
    # --- 1. Load Environment Variables & Data ---
    load_dotenv()
    print(f"Loading movie plots from '{csv_path}'...")
    try:
        df = pd.read_csv(csv_path)
        df.dropna(subset=['Plot'], inplace=True)
        print(f"Successfully loaded {len(df)} movies.")
    except FileNotFoundError:
        print(f"Error: The file '{csv_path}' was not found.")
        return

    # --- 2. Initialize ChromaDB Client and Embedding Model ---
    print("Initializing ChromaDB client and Sentence Transformer model...")
    chroma_api_key = os.getenv('CHROMA_API_KEY')
    chroma_tenant = os.getenv('CHROMA_TENANT')
    chroma_database = os.getenv('CHROMA_DATABASE')

    if not all([chroma_api_key, chroma_tenant, chroma_database]):
        print("Error: ChromaDB credentials not found in .env file.")
        print("Please ensure your .env file has CHROMA_API_KEY, CHROMA_TENANT, and CHROMA_DATABASE.")
        return

    try:
        # Use the CloudClient to connect to your ChromaDB Cloud instance
        client = chromadb.CloudClient(
            tenant=chroma_tenant,
            database=chroma_database,
            api_key=chroma_api_key
        )
        print(f"Successfully connected to ChromaDB Cloud.")
    except Exception as e:
        print(f"Failed to connect to ChromaDB: {e}")
        return

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print("Initialization complete.")

    # --- 3. Create or Recreate ChromaDB Collection ---
    print(f"Setting up collection: '{collection_name}'")
    
    # To ensure a clean run, delete the collection if it exists
    try:
        if collection_name in [c.name for c in client.list_collections()]:
            print(f"Collection '{collection_name}' already exists. Deleting it for a fresh start.")
            client.delete_collection(name=collection_name)
    except Exception as e:
        print(f"Error deleting collection: {e}")
        return

    # Create a new collection
    collection = client.get_or_create_collection(name=collection_name)
    print(f"Collection '{collection_name}' is ready.")

    # --- 4. Prepare Data for ChromaDB ---
    print("Preparing documents, metadata, and IDs...")
    documents = []
    metadatas = []
    ids = []
    
    # Use iterrows to get a unique index for each row
    for index, row in df.iterrows():
        title = row['Title']
        plot = row['Plot']
        plot_chunks = chunk_text(plot)
        for i, chunk in enumerate(plot_chunks):
            documents.append(chunk)
            metadatas.append({'title': str(title), 'chunk_index': i})
            # ChromaDB requires a unique string ID for each document
            ids.append(f"id_{index}_{i}")
    
    if not documents:
        print("No documents to process. Exiting.")
        return

    # --- 5. Generate Embeddings ---
    print("Generating embeddings...")
    embeddings = model.encode(documents, show_progress_bar=True, normalize_embeddings=True)
    print(f"Generated {len(embeddings)} embeddings.")

    # --- 6. Batch Store Data in ChromaDB ---
    print("Storing data in ChromaDB. This may take a while...")
    
    # ChromaDB's .add() method handles batching automatically.
    # We provide all documents, embeddings, metadata, and IDs at once.
    try:
        collection.add(
            embeddings=embeddings.tolist(), # Embeddings need to be a list
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
    except Exception as e:
        print(f"An error occurred during data insertion: {e}")
        return
        
    print(f"\n--- Processing Complete ---")
    
    # Verify the number of objects using collection.count()
    count = collection.count()
    print(f"Total documents successfully stored: {count}")


if __name__ == '__main__':
    CSV_FILE_PATH = 'subset_movie_plots.csv'
    # It's good practice to use snake_case for collection names in ChromaDB
    COLLECTION_NAME = 'movie_plots' 
    
    process_and_embed_data(CSV_FILE_PATH, COLLECTION_NAME)