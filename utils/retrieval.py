import faiss
import pickle
import numpy as np
import os

client = None

# --- Load FAISS index and metadata from project root ---
index_path = "trip_safe_index.faiss"
metadata_path = "trip_safe_metadata.pkl"

if os.path.exists(index_path) and os.path.exists(metadata_path):
    index = faiss.read_index(index_path)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
else:
    raise FileNotFoundError("FAISS index or metadata not found. Please ensure both files exist at project root.")

def set_openai_api_key(api_key: str):
    global client
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

def get_embedding(text: str) -> list[float]:
    if client is None:
        raise ValueError("OpenAI client not initialized. Call set_openai_api_key() first.")
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def query_faiss(query, k=20):
    query_embedding = get_embedding(query)
    distances, indices = index.search(
        np.array([query_embedding]).astype('float32'), k
    )
    results = []
    for i, idx in enumerate(indices[0]):
        if idx == -1:
            continue
        results.append({
            "text": metadata["texts"][idx],
            "source": metadata["sources"][idx],
            "distance": distances[0][i]
        })
    return results

def generate_answer(query, results, conversation_context=""):
    context = "\n".join([
        f"Source: {r['source']}\nText: {r['text']}" for r in results
    ])
    prompt = f"""Given the following documents and recent conversation, answer the query.

Conversation:
{conversation_context}

Documents:
{context}

Query:
{query}

Answer:"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are an expert travel insurance sales pitcher of TripSafe. You should answer questions based on provided documents, answer in detail, and guide the customer with correct steps and information."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=500,
        temperature=0.3
    )
    return response.choices[0].message.content.strip()
