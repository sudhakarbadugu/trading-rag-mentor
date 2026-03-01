"""
app.py
------
Main Streamlit application for the Trading RAG Mentor.
Provides the user interface, session management, and coordinates the RAG pipeline.
"""

import os
import logging
from pathlib import Path

import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.language_models import FakeListLLM
from dotenv import load_dotenv

from build_index import build_index
from prompts import TRADING_MENTOR_PROMPT, REFORMULATION_PROMPT
from chat_history import init_db, load_messages, save_message, clear_messages, list_sessions
from retrieval import build_bm25_retriever, hybrid_retrieve_and_rerank
import tiktoken

logging.basicConfig(level=logging.INFO)
load_dotenv()

# ── Ensure SQLite schema exists ───────────────────────────────────────────────
init_db()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="My Trading RAG Coach", page_icon="📈", layout="wide")

# ── Token Encoding (cached once) ──────────────────────────────────────────────
@st.cache_resource
def get_encoder():
    """
    Initialize the tiktoken encoder for cl100k_base.
    
    Returns:
        Encoding: The tiktoken encoding object.
    """
    return tiktoken.get_encoding("cl100k_base")

encoder = get_encoder()
TOKEN_LIMIT = 6000  # Safe threshold leaving ~2k tokens for the generation

st.title("📈 My Personal Trading RAG Coach")
st.caption("Ask anything about my momentum & price action video transcripts")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ RAG Configuration")

    # ── Session management ────────────────────────────────────────────────────
    st.subheader("💬 Chat Session")

    existing_sessions = list_sessions()

    if "session_id" not in st.session_state:
        st.session_state.session_id = existing_sessions[0] if existing_sessions else "session_1"

    if st.button("➕ New Session", use_container_width=True):
        all_ids = list_sessions()
        nums = []
        for sid in all_ids:
            try:
                nums.append(int(sid.replace("session_", "")))
            except ValueError:
                pass
        next_num = (max(nums) + 1) if nums else 1
        st.session_state.session_id = f"session_{next_num}"
        st.session_state.messages = []
        st.rerun()

    if existing_sessions:
        chosen = st.selectbox(
            "Switch session",
            options=existing_sessions,
            index=(
                existing_sessions.index(st.session_state.session_id)
                if st.session_state.session_id in existing_sessions
                else 0
            ),
            key="session_selector",
        )
        if chosen != st.session_state.session_id:
            st.session_state.session_id = chosen
            st.session_state.messages = []
            st.rerun()

    if st.button("🗑️ Clear This Session", use_container_width=True):
        clear_messages(st.session_state.session_id)
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # ── Model selector ────────────────────────────────────────────────────────
    st.subheader("🤖 Model")
    _env_model = os.environ.get("GROQ_MODEL_NAME", "openai/gpt-oss-120b")
    AVAILABLE_MODELS = [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "openai/gpt-oss-120b",
    ]
    if _env_model not in AVAILABLE_MODELS:
        AVAILABLE_MODELS.insert(0, _env_model)
    selected_model = st.selectbox(
        "Groq Model",
        options=AVAILABLE_MODELS,
        index=AVAILABLE_MODELS.index(_env_model),
        help="Choose the Groq LLM to use for answering questions.",
    )

    # ── Temperature ───────────────────────────────────────────────────────────
    st.subheader("🌡️ Temperature")
    temperature = st.slider(
        "Temperature",
        min_value=0.0, max_value=1.0, value=0.0, step=0.05,
        help="Higher = more creative, Lower = more deterministic.",
    )

    # ── Retrieval k ───────────────────────────────────────────────────────────
    st.subheader("🔍 Retrieval")
    k_value = st.slider(
        "Number of chunks to retrieve (k)",
        min_value=1, max_value=15, value=6, step=1,
        help="How many document chunks are retrieved per query.",
    )
    use_hybrid = st.toggle(
        "🔀 Hybrid Search (BM25 + Vector)",
        value=True,
        help="Combines keyword matching (BM25) with semantic search for better recall.",
    )
    use_rerank = st.toggle(
        "📊 Cross-Encoder Re-ranking",
        value=True,
        help="Re-scores retrieved chunks with a cross-encoder model for higher precision.",
    )

    st.divider()

    # ── Conversational memory toggle ──────────────────────────────────────
    st.subheader("🧠 Memory")
    use_memory = st.toggle(
        "Conversational Memory",
        value=True,
        help="When ON, follow-up questions are rewritten as standalone queries "
             "using recent chat context. Turn OFF for independent questions.",
    )
    memory_turns = st.slider(
        "History turns",
        min_value=1, max_value=10, value=5,
        help="Number of recent Q&A turns to use for context.",
    ) if use_memory else 0

    st.divider()
    st.caption(
        "**Model** controls which LLM answers your question.  \n"
        "**Temperature** controls creativity.  \n"
        "**k** controls how many transcript chunks are searched.  \n"
        "**Hybrid Search** enables BM25 + vector retrieval.  \n"
        "**Re-ranking** uses a cross-encoder for higher precision.  \n"
        "**Memory** enables follow-up question understanding."
    )

# ── Build index if needed ─────────────────────────────────────────────────────
try:
    build_index()
except Exception as e:
    st.error(f"Failed to check or build the document index: {e}")

# ── Load vector DB (cached) ───────────────────────────────────────────────────
@st.cache_resource
def load_db():
    """
    Load the ChromaDB vectorstore from Disk.
    
    Returns:
        Chroma: The persistent vectorstore instance.
    """
    ROOT_DIR = Path(__file__).resolve().parent.parent
    DB_DIR = ROOT_DIR / "data" / "chroma_db"
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory=str(DB_DIR), embedding_function=embeddings)

try:
    vectorstore = load_db()
except Exception as e:
    st.error(f"Failed to load the vector database: {e}")
    vectorstore = None

# ── Build BM25 retriever (cached) ───────────────────────────────────────────
@st.cache_resource
def load_bm25(_vectorstore):
    """
    Initialize the BM25 retriever from the vectorstore's documents.
    
    Args:
        _vectorstore (Chroma): The vectorstore instance.
        
    Returns:
        BM25Retriever: The configured BM25 retriever.
    """
    if _vectorstore is None:
        return None
    return build_bm25_retriever(_vectorstore)

bm25_retriever = load_bm25(vectorstore) if vectorstore else None

# ── Build LLM ─────────────────────────────────────────────────────────────────
prompt = PromptTemplate.from_template(TRADING_MENTOR_PROMPT)
groq_api_key = os.environ.get("GROQ_API_KEY")

if groq_api_key:
    llm = ChatGroq(
        temperature=temperature,
        model_name=selected_model,
        groq_api_key=groq_api_key,
        streaming=True,
    )
else:
    llm = FakeListLLM(
        responses=["Error: GROQ_API_KEY not set. Please add it to your .env file."]
    )

# ── Load / sync chat history from SQLite ─────────────────────────────────────
session_id = st.session_state.session_id
if "messages" not in st.session_state or not st.session_state.messages:
    st.session_state.messages = load_messages(session_id)

# ── Render existing conversation ──────────────────────────────────────────────
st.caption(f"🗂️ Session: `{session_id}`")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input & response ─────────────────────────────────────────────────────
if query := st.chat_input("Ask any question about my momentum & price action videos?"):
    save_message(session_id, "user", query)
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        if vectorstore is None:
            st.error("Vector database not loaded. Please check the error messages above.")
        else:
            try:
                # ── Step 0: Build chat history for context ────────────────────
                chat_history_str = ""
                if use_memory and len(st.session_state.messages) > 1:
                    # Grab last N turns (each turn = user + assistant)
                    recent = st.session_state.messages[-(memory_turns * 2):][:-1]  # exclude current query
                    lines = []
                    for msg in recent:
                        role = "User" if msg["role"] == "user" else "Mentor"
                        # Truncate long answers to keep prompt concise
                        content = msg["content"][:500]
                        lines.append(f"{role}: {content}")
                    chat_history_str = "\n".join(lines)

                # ── Step 1: Reformulate query if memory is on ─────────────────
                search_query = query
                if use_memory and chat_history_str:
                    with st.spinner("Understanding your follow-up..."):
                        reformulation_prompt = REFORMULATION_PROMPT.format(
                            chat_history=chat_history_str, question=query
                        )
                        reformulated = llm.invoke(reformulation_prompt).content.strip()
                        # Only use reformulated if it differs meaningfully
                        if reformulated and reformulated != query:
                            search_query = reformulated
                            st.info(f"🔄 Searching for: *{search_query}*", icon="🧠")
                            logging.info(f"Reformulated: '{query}' → '{search_query}'")

                # ── Step 2: Retrieve relevant chunks + scores ─────────────────
                with st.spinner(
                    "Searching my notes..."
                    + (" (hybrid + re-ranking)" if use_hybrid and use_rerank else "")
                ):
                    scored_docs = hybrid_retrieve_and_rerank(
                        query=search_query,
                        vectorstore=vectorstore,
                        bm25_retriever=bm25_retriever,
                        k=k_value,
                        use_hybrid=use_hybrid,
                        use_rerank=use_rerank,
                    )
                    source_docs = [doc for doc, _ in scored_docs]
                    score_map = {
                        doc.page_content: round(score, 3) for doc, score in scored_docs
                    }

                retrieval_mode = []
                if use_hybrid:
                    retrieval_mode.append("hybrid")
                if use_rerank:
                    retrieval_mode.append("re-ranked")
                mode_str = " + ".join(retrieval_mode) if retrieval_mode else "vector"
                logging.info(f"\n🔎 [{mode_str}] Retrieved {len(source_docs)} chunks for: '{search_query}'")
                for i, (doc, score) in enumerate(scored_docs):
                    logging.info(f"--- Doc {i+1} | Score: {score:.3f} | {doc.metadata} ---")
                    logging.info(f"Content: {doc.page_content[:200]}\n")

                # ── Step 3: Format context + prompt (with token limits) ───────
                history_block = chat_history_str if chat_history_str else "(No prior conversation)"
                
                dropped_chunks = 0
                while True:
                    context = "\n\n".join(doc.page_content for doc in source_docs)
                    formatted_prompt = prompt.format(
                        context=context, question=query, chat_history=history_block
                    )
                    
                    # Estimate tokens
                    token_count = len(encoder.encode(formatted_prompt))
                    
                    if token_count <= TOKEN_LIMIT or len(source_docs) == 0:
                        break
                        
                    # Truncate lowest-scoring chunk (last element) and check again
                    dropped_doc = source_docs.pop()
                    scored_docs = [(d, s) for d, s in scored_docs if d != dropped_doc]
                    dropped_chunks += 1
                    
                if dropped_chunks > 0:
                    logging.warning(f"⚠️ Truncated {dropped_chunks} low-scoring chunk(s) to pass {TOKEN_LIMIT}-token limit.")
                    st.warning(f"⚠️ Dropped the {dropped_chunks} lowest-scoring chunk(s) from memory to fit context limits.", icon="✂️")
                    
                # ── Step 4: Stream response via st.write_stream ───────────────
                def stream_tokens():
                    for chunk in llm.stream(formatted_prompt):
                        yield chunk.content

                answer = st.write_stream(stream_tokens())

                # ── Step 5: Source citations with relevance badges ────────────
                if source_docs:
                    # Deduplicate by filename, keeping highest-score chunk per file
                    seen: dict = {}
                    for doc in source_docs:
                        src = doc.metadata.get("source", "Unknown source")
                        score = score_map.get(doc.page_content, 0.0)
                        if src not in seen or score > seen[src][1]:
                            seen[src] = (doc, score)

                    unique_sources = list(seen.values())

                    with st.expander(
                        f"📄 Sources ({len(unique_sources)} "
                        f"transcript{'s' if len(unique_sources) > 1 else ''})"
                    ):
                        for i, (doc, score) in enumerate(unique_sources):
                            src = doc.metadata.get("source", "Unknown source")
                            page = doc.metadata.get("page")
                            filename = src.split("/")[-1]

                            pct = int(score * 100)
                            badge = (
                                f"🟢 {pct}% match" if pct >= 70
                                else f"🟡 {pct}% match" if pct >= 40
                                else f"🔴 {pct}% match"
                            )

                            label = f"**{i+1}. {filename}**"
                            if page is not None:
                                label += f" — Page {page + 1}"

                            col1, col2 = st.columns([4, 1])
                            with col1:
                                st.markdown(label)
                            with col2:
                                st.markdown(badge)

                            st.caption(
                                doc.page_content[:300].strip()
                                + ("…" if len(doc.page_content) > 300 else "")
                            )
                            if i < len(unique_sources) - 1:
                                st.divider()

                # ── Step 6: Persist answer to session history ─────────────────
                save_message(session_id, "assistant", answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                st.error(f"An error occurred while answering your question: {e}")