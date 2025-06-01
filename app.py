import streamlit as st
import uuid
from utils.retrieval import query_faiss, generate_answer, set_openai_api_key
from utils.helpers import load_chat, save_chat, list_chats

# Initialize OpenAI client with key from Streamlit secrets
set_openai_api_key(st.secrets["api_keys"]["openai"])

st.set_page_config(page_title="TripSafe Agent", layout="wide")

# Because we will run “cd trip_safe_chatbot && streamlit run app.py”,
# Streamlit’s working directory is “trip_safe_chatbot/”. So the logo path is correct:
st.image("TripSafe logo-01.png", width=220)
st.title("🧳 TripSafe AI Assistant")

# Initialize session state
if "chat_id" not in st.session_state:
    st.session_state.chat_id = str(uuid.uuid4())
    st.session_state.messages = []

# Sidebar – Chat history
st.sidebar.header("🕘 Chat History")
all_chats = list_chats()
for cid, meta in all_chats.items():
    if st.sidebar.button(meta["title"], key=cid):
        st.session_state.chat_id = cid
        st.session_state.messages = load_chat(cid)

# New Chat Button
if st.sidebar.button("➕ Start New Chat"):
    st.session_state.chat_id = str(uuid.uuid4())
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_query = st.chat_input("Ask about TripSafe travel insurance...")
if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    top_k_docs = query_faiss(user_query)

    # Get last 5 messages for context
    last_msgs = st.session_state.messages[-5:]
    conversation_context = "\n".join([f"{m['role']}: {m['content']}" for m in last_msgs])

    response = generate_answer(user_query, top_k_docs, conversation_context)
    st.session_state.messages.append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        st.markdown(response)

    save_chat(st.session_state.chat_id, st.session_state.messages)
