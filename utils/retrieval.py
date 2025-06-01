import openai
import faiss
import pickle
import numpy as np

# At runtime, this code will do:
#    index = faiss.read_index("trip_safe_index.faiss")
#    with open("trip_safe_metadata.pkl", "rb") as f:
#        metadata = pickle.load(f)

faiss_path = "trip_safe_index.faiss"
meta_path  = "trip_safe_metadata.pkl"

index = faiss.read_index(faiss_path)
with open(meta_path, "rb") as f:
    metadata = pickle.load(f)

def get_embedding(text: str) -> list[float]:
    response = openai.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    )
    return response["data"][0]["embedding"]


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
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert travel insurance sales pitcher of TripSafe. You should answer questions based on provided documents, answer in detail, and guide the customer with correct steps and information."},
            {"role": "user",   "content": prompt}
        ],
        max_tokens=500,
        temperature=0.3
    )
    return response['choices'][0]['message']['content'].strip()
