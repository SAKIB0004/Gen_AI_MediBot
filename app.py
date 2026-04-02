import time
import streamlit as st
from langchain_pinecone import PineconeVectorStore
from langchain_core.chat_history import InMemoryChatMessageHistory

from src.config import validate_env, PINECONE_INDEX_NAME, PINECONE_NAMESPACE
from src.embeddings import get_embedder
from src.rag_groq import build_groq_rag_chain


# ----------------------------
# FIXED SETTINGS (no sidebar)
# ----------------------------
TOP_K = 5
SCORE_THRESHOLD = 0.25
SHOW_DEBUG = False  # set True if you want to see retrieved chunks

# Memory tuning
MAX_HISTORY_MESSAGES = 8  # last 8 messages = last 4 user/assistant turns

# Micro-interaction tuning
TYPING_PAUSE_SEC = 0.6          # "thinking..." pause before streaming
STREAM_DELAY_SEC = 0.012        # character streaming speed


# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="MediBot", page_icon="🩺", layout="centered")


# ----------------------------
# Global UI styles
# ----------------------------
st.markdown(
    """
    <style>
      /* Make the whole app feel cleaner */
      .block-container { padding-top: 1.2rem; padding-bottom: 3rem; }
      h1, h2, h3 { letter-spacing: -0.02em; }

      /* Chat bubbles slight polish */
      .stChatMessage { border-radius: 14px; }
      .stChatMessage [data-testid="stMarkdownContainer"] { line-height: 1.55; }

      /* Buttons / inputs */
      .stTextInput, .stChatInput { border-radius: 14px; }

      /* Subtle divider */
      .soft-divider {
        height: 1px; width: 100%;
        background: linear-gradient(90deg, transparent, rgba(148,163,184,.35), transparent);
        margin: 14px 0 18px 0;
      }

      /* Header card */
      .medibot-header {
        background: radial-gradient(1200px 420px at 20% 10%, rgba(56,189,248,0.16), transparent 60%),
                    linear-gradient(135deg, #0b1220, #020617);
        border: 1px solid rgba(148,163,184,0.18);
        border-radius: 18px;
        padding: 20px 22px;
        box-shadow: 0 10px 30px rgba(2,6,23,0.55);
        margin-bottom: 14px;
      }

      .medibot-title {
        font-size: 1.5rem;
        font-weight: 800;
        margin: 0;
        color: #e5e7eb;
        display: flex;
        align-items: center;
        gap: 10px;
      }

      .medibot-subtitle {
        margin-top: 6px;
        font-size: 0.95rem;
        color: rgba(226,232,240,0.75);
      }

      .badge-row {
        margin-top: 10px;
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
      }

      .badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 10px;
        border-radius: 999px;
        font-size: 0.75rem;
        border: 1px solid rgba(56,189,248,0.25);
        background: rgba(56,189,248,0.10);
        color: #7dd3fc;
      }

      .badge2 {
        border: 1px solid rgba(167,139,250,0.25);
        background: rgba(167,139,250,0.10);
        color: #c4b5fd;
      }

      .badge3 {
        border: 1px solid rgba(34,197,94,0.22);
        background: rgba(34,197,94,0.10);
        color: #86efac;
      }

      /* Safety card */
      .safety-card {
        padding: 14px 16px;
        border-radius: 16px;
        background: rgba(15, 23, 42, 0.55);
        border: 1px solid rgba(148,163,184,0.18);
        color: rgba(226,232,240,0.86);
        font-size: 0.92rem;
        margin-bottom: 14px;
      }

      /* Tiny helper chips under input */
      .hint-row {
        margin-top: 10px;
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
      }

      .hint {
        padding: 6px 10px;
        border-radius: 999px;
        border: 1px solid rgba(148,163,184,0.18);
        background: rgba(2,6,23,0.35);
        color: rgba(226,232,240,0.70);
        font-size: 0.78rem;
      }

      /* Source list */
      .source-list li { margin-bottom: 6px; }

      /* Top actions */
      .top-actions {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 8px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------
# Creative header
# ----------------------------
st.markdown(
    """
    <div class="medibot-header">
      <h2 class="medibot-title">🩺 MediBot</h2>
      <div class="medibot-subtitle">Medical Book Question Answering • Pinecone + Groq</div>
      <div class="badge-row">
        <span class="badge">📚 Source-grounded</span>
        <span class="badge badge2">🔎 RAG Retrieval</span>
        <span class="badge badge3">🧠 Conversation Memory</span>
      </div>
    </div>
    <div class="soft-divider"></div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="safety-card">
      ⚠️ <b>Disclaimer:</b> Answers are generated strictly from the uploaded medical book content (retrieved chunks).
      This application does not provide medical advice.
    </div>
    """,
    unsafe_allow_html=True,
)


# ----------------------------
# Validate env
# ----------------------------
validate_env()


# ----------------------------
# Retriever (namespace-safe)
# ----------------------------
def build_retriever(k: int):
    embedder = get_embedder()
    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embedder,
        namespace=PINECONE_NAMESPACE,
    )

    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": max(20, k * 4)},
    )


@st.cache_resource
def load_rag():
    retriever = build_retriever(TOP_K)
    chain = build_groq_rag_chain(retriever)
    return retriever, chain


retriever, rag_chain = load_rag()


# ----------------------------
# Chat state
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = InMemoryChatMessageHistory()

    # rebuild memory from UI messages if they already exist
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.session_state.chat_history.add_user_message(msg["content"])
        elif msg["role"] == "assistant":
            st.session_state.chat_history.add_ai_message(msg["content"])


# ----------------------------
# Clear chat
# ----------------------------
col1, col2 = st.columns([8, 1])
with col2:
    if st.button("🗑️", help="Clear chat"):
        st.session_state.messages = []
        st.session_state.chat_history = InMemoryChatMessageHistory()
        st.rerun()


# ----------------------------
# Render old messages
# ----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ----------------------------
# UX: Streaming helper
# ----------------------------
def stream_markdown(text: str, delay: float = STREAM_DELAY_SEC):
    """
    Progressive streaming effect (ChatGPT-like).
    Streams markdown safely as plain text progressively.
    """
    buf = ""
    placeholder = st.empty()
    for ch in text:
        buf += ch
        placeholder.markdown(buf)
        time.sleep(delay)
    return placeholder


# ----------------------------
# Memory helpers
# ----------------------------
def get_recent_history_text(max_messages: int = MAX_HISTORY_MESSAGES) -> str:
    """
    Convert InMemoryChatMessageHistory into a compact plain-text history
    for retrieval and answer generation.
    """
    recent_messages = st.session_state.chat_history.messages[-max_messages:]
    lines = []

    for msg in recent_messages:
        if msg.type == "human":
            role = "User"
        elif msg.type == "ai":
            role = "Assistant"
        else:
            role = msg.type.capitalize()

        lines.append(f"{role}: {msg.content}")

    return "\n".join(lines)


def build_memory_augmented_query(question: str) -> str:
    """
    Use recent chat history only to resolve references like:
    'it', 'that', 'this', 'previous result', etc.
    """
    history_text = get_recent_history_text()

    if not history_text.strip():
        return question

    return f"""
Use the previous conversation only to understand follow-up references.
Do not answer the history itself.

Previous conversation:
{history_text}

Current user question:
{question}
""".strip()


# ----------------------------
# Strong anti-hallucination guard + memory-aware answer
# ----------------------------
def safe_answer(question: str):
    """
    1) Build memory-aware query from recent chat history
    2) Retrieve docs directly (with scores)
    3) If weak/no docs -> "I don't know"
    4) Else run RAG chain
    """
    embedder = get_embedder()
    vs = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embedder,
        namespace=PINECONE_NAMESPACE,
    )

    enriched_question = build_memory_augmented_query(question)
    docs_with_scores = vs.similarity_search_with_score(enriched_question, k=TOP_K)

    if not docs_with_scores:
        return "I don't know based on the provided book.", [], docs_with_scores

    numeric_scores = [s for _, s in docs_with_scores if isinstance(s, (int, float))]
    if numeric_scores and max(numeric_scores) < SCORE_THRESHOLD:
        return "I don't know based on the provided book.", [], docs_with_scores

    out = rag_chain.invoke(enriched_question)

    answer_msg = out.get("answer")
    answer_text = answer_msg.content if hasattr(answer_msg, "content") else str(answer_msg)

    docs = out.get("docs", [])
    return answer_text, docs, docs_with_scores


# ----------------------------
# Input
# ----------------------------
user_q = st.chat_input("Ask a question from the medical book...")

st.markdown(
    """
    <div class="hint-row">
      <div class="hint">Try: “What is hypertension?”</div>
      <div class="hint">Try: “Symptoms of diabetes?”</div>
      <div class="hint">Try: “Explain anemia in simple words”</div>
    </div>
    """,
    unsafe_allow_html=True,
)


if user_q:
    # Show/store user message in UI
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    # ----------------------------
    # Assistant response
    # ----------------------------
    with st.chat_message("assistant"):
        typing_placeholder = st.empty()
        typing_placeholder.markdown("⌛ *Thinking…*")

        time.sleep(TYPING_PAUSE_SEC)

        with st.spinner("Searching the book..."):
            answer, docs, docs_with_scores = safe_answer(user_q)

        typing_placeholder.empty()
        stream_markdown(answer)

    # Store final assistant message in UI
    st.session_state.messages.append({"role": "assistant", "content": answer})

    # Store turn in real chat memory
    st.session_state.chat_history.add_user_message(user_q)
    st.session_state.chat_history.add_ai_message(answer)

    # Sources
    if docs:
        with st.expander("Sources"):
            seen = set()
            items = []

            for d in docs:
                src = d.metadata.get("source", "Unknown source")
                page = d.metadata.get("page", "N/A")
                key = (src, page)

                if key in seen:
                    continue

                seen.add(key)
                items.append(f"<li><b>{src}</b> — page <code>{page}</code></li>")

            st.markdown(
                f"<ul class='source-list'>{''.join(items)}</ul>",
                unsafe_allow_html=True,
            )

    # Debug retrieved context
    if SHOW_DEBUG and docs_with_scores:
        with st.expander("Debug: Retrieved chunks"):
            for i, (d, s) in enumerate(docs_with_scores, start=1):
                st.markdown(f"**Chunk {i}** | score: `{s}`")
                st.code((d.page_content or "")[:1500])