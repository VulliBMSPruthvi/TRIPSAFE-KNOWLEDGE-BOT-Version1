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




def generate_answer(query, results, conversation_history=None):
    """
    conversation_history: list of dicts like:
    [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
        ...
    ]
    It should contain up to the last 5 user-assistant exchanges.
    """

    # Format documents context
    documents_text = "\n".join([
        f"Source: {r['source']}\nText: {r['text']}" for r in results
    ])

    # Prepare conversation context text from last 5 messages (if any)
    conversation_text = ""
    if isinstance(conversation_history, list):
        try:
            last_msgs = conversation_history[-5:]
            conversation_text = "\n".join([
                f"{msg['role'].capitalize()}: {msg['content']}"
                for msg in last_msgs if isinstance(msg, dict) and 'role' in msg and 'content' in msg
            ])
        except Exception as e:
            print(f"Error formatting conversation history: {e}")
            conversation_text = ""

    # Build user prompt with conversation context and documents
    user_prompt = f"""Given the following documents and recent conversation, answer the query.

Conversation:
{conversation_text}

Documents:
{documents_text}

Query:
{query}

Answer:"""

    system_message = {
        "role": "system",
        "content": (
            "You are supposed to be polite, specific and clear while answering. "
            "You are an expert travel insurance sales representative for TripSafe, a trusted travel insurance provider. "
            "Use the provided documents and recent conversation context to answer customer queries accurately and in detail. "
            "Always provide clear, actionable guidance tailored to the customer’s needs. "
            "Avoid speculation or unrelated information; base all answers strictly on the given documents. "
            "Be polite, professional, and helpful."
        )
    }

    user_message = {
        "role": "user",
        "content": user_prompt
    }

    messages = [system_message, user_message]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=700,
        temperature=0.25
    )

    return response.choices[0].message.content.strip()
