import os
import json

CHAT_DIR = os.path.join(os.path.dirname(__file__), "..", "chats")
os.makedirs(CHAT_DIR, exist_ok=True)

def save_chat(chat_id, messages):
    with open(os.path.join(CHAT_DIR, f"{chat_id}.json"), "w") as f:
        json.dump(messages, f)

def load_chat(chat_id):
    try:
        with open(os.path.join(CHAT_DIR, f"{chat_id}.json"), "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def list_chats():
    chats = {}
    for fname in os.listdir(CHAT_DIR):
        if fname.endswith(".json"):
            chat_id = fname.replace(".json", "")
            with open(os.path.join(CHAT_DIR, fname)) as f:
                messages = json.load(f)
                # Pick the first user message as the chat title (or "Untitled")
                title = next((m["content"] for m in messages if m["role"] == "user"), "Untitled")
                # **Use a single-dict literal here:**
                chats[chat_id] = {
                    "title": title[:40] + ("..." if len(title) > 40 else "")
                }
    return chats
