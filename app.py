import streamlit as st
import asyncio
import uuid
import tempfile
import os
from datetime import datetime
from typing import List, Dict, Optional, Set

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, Filter, FieldCondition, MatchAny
from fastembed import TextEmbedding
import ollama
import edge_tts

# ── Windows asyncio fix ───────────────────────────────────────────
import sys
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

COLLECTION_NAME = "docs_embeddings"

# ══════════════════════════════════════════════════════════════════
# SESSION STATE
# ── NEW: chat_history, doc_registry added
# ── CHANGED: setup_complete now auto-resolves if Qdrant has docs
# ══════════════════════════════════════════════════════════════════

def init_session_state():
    defaults = {
        "client": None,
        "embedding_model": None,
        "ollama_model": "llama3.2",
        "tts_voice": "en-US-JennyNeural",

        # NEW — persists all Q&A pairs for the session
        "chat_history": [],

        # NEW — {filename: {chunks, uploaded_at}} for THIS session's uploads
        # Qdrant-persisted docs from prior sessions are discovered separately
        "doc_registry": {},

        # NEW — set of filenames the user has selected to query against
        "selected_sources": set(),

        # Whether Qdrant connection + embedding model are ready
        "setup_complete": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════
# QDRANT — unchanged logic, same as before
# ══════════════════════════════════════════════════════════════════

def setup_qdrant():
    client = QdrantClient(url="http://localhost:6333")
    embedding_model = TextEmbedding()
    test_emb = list(embedding_model.embed(["test"]))[0]
    dim = len(test_emb)
    try:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )
    except Exception as e:
        if "already exists" not in str(e):
            raise e
    return client, embedding_model


# ══════════════════════════════════════════════════════════════════
# NEW — Discover all unique source names stored in Qdrant
# Scrolls through all points and collects distinct "source" payloads.
# This is how we surface docs from previous sessions.
# ══════════════════════════════════════════════════════════════════

def get_all_sources_from_qdrant(client: QdrantClient) -> List[Dict]:
    """
    Returns a list of dicts: {source, chunk_count, uploaded_at}
    by scrolling through all Qdrant points.
    """
    sources: Dict[str, Dict] = {}
    offset = None

    while True:
        result, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=250,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )
        for point in result:
            payload = point.payload or {}
            src = payload.get("source", "unknown")
            if src not in sources:
                sources[src] = {
                    "source": src,
                    "chunk_count": 0,
                    "uploaded_at": payload.get("crawl_date", "unknown")
                }
            sources[src]["chunk_count"] += 1

        if next_offset is None:
            break
        offset = next_offset

    # Sort: most recently uploaded first
    return sorted(sources.values(), key=lambda x: x["uploaded_at"], reverse=True)


# ══════════════════════════════════════════════════════════════════
# FILE PARSING — unchanged
# ══════════════════════════════════════════════════════════════════

def parse_uploaded_file(uploaded_file) -> str:
    filename = uploaded_file.name.lower()
    raw = uploaded_file.read()

    if filename.endswith(".txt") or filename.endswith(".md"):
        return raw.decode("utf-8", errors="ignore")

    elif filename.endswith(".pdf"):
        try:
            import pypdf, io
            reader = pypdf.PdfReader(io.BytesIO(raw))
            return "\n\n".join(p.extract_text() for p in reader.pages if p.extract_text())
        except ImportError:
            st.error("Install pypdf: pip install pypdf")
            return ""

    elif filename.endswith(".html") or filename.endswith(".htm"):
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(raw, "html.parser")
            for tag in soup(["script", "style", "nav", "footer"]):
                tag.decompose()
            return soup.get_text(separator="\n", strip=True)
        except ImportError:
            import re
            return re.sub(r"<[^>]+>", " ", raw.decode("utf-8", errors="ignore"))

    elif filename.endswith(".docx"):
        try:
            import docx, io
            doc = docx.Document(io.BytesIO(raw))
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except ImportError:
            st.error("Install python-docx: pip install python-docx")
            return ""

    return raw.decode("utf-8", errors="ignore")


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return [c for c in chunks if c.strip()]


# ══════════════════════════════════════════════════════════════════
# STORE EMBEDDINGS — unchanged logic
# CHANGED: returns chunk count per file so we can update doc_registry
# ══════════════════════════════════════════════════════════════════

def store_embeddings(client, embedding_model, pages: List[Dict]) -> Dict[str, int]:
    """Returns {source_name: chunk_count}"""
    counts: Dict[str, int] = {}
    for page in pages:
        content = page.get("content", "")
        if not content.strip():
            continue
        chunks = chunk_text(content)
        src = page.get("source", "unknown")
        counts[src] = 0
        for i, chunk in enumerate(chunks):
            embedding = list(embedding_model.embed([chunk]))[0]
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=[models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding.tolist(),
                    payload={
                        "content": chunk,
                        "source": src,
                        "chunk_index": i,
                        "crawl_date": datetime.now().isoformat()
                    }
                )]
            )
            counts[src] += 1
    return counts


# ══════════════════════════════════════════════════════════════════
# OLLAMA QUERY — unchanged
# ══════════════════════════════════════════════════════════════════

def query_ollama(model: str, context: str, question: str) -> str:
    prompt = f"""You are a helpful documentation assistant.

Based on the following documentation content, answer the user's question clearly and concisely.
Write your response so it sounds natural when spoken aloud — avoid markdown formatting like bullet points or headers.
Use plain conversational sentences.

Documentation:
{context}

Question: {question}

Answer:"""
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]


# ══════════════════════════════════════════════════════════════════
# TTS — unchanged
# ══════════════════════════════════════════════════════════════════

async def _generate_tts_async(text: str, voice: str) -> str:
    temp_path = os.path.join(tempfile.gettempdir(), f"tts_{uuid.uuid4()}.mp3")
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(temp_path)
    return temp_path

def generate_tts(text: str, voice: str) -> str:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_generate_tts_async(text, voice))
    finally:
        loop.close()


# ══════════════════════════════════════════════════════════════════
# QUERY PIPELINE
# CHANGED: now accepts selected_sources to filter Qdrant results.
# If selected_sources is empty → query all docs (same as before).
# If non-empty → adds a Qdrant Filter so only matching sources
#   are searched. This is the key change enabling per-doc querying.
# ══════════════════════════════════════════════════════════════════

def process_query(query: str, selected_sources: Optional[Set[str]] = None) -> Dict:
    try:
        client = st.session_state.client
        embedding_model = st.session_state.embedding_model

        query_emb = list(embedding_model.embed([query]))[0]

        # NEW — build a Qdrant filter if specific sources are selected
        query_filter = None
        if selected_sources:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="source",
                        match=MatchAny(any=list(selected_sources))
                    )
                ]
            )

        search_resp = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_emb.tolist(),
            query_filter=query_filter,   # NEW — scoped search
            limit=4,
            with_payload=True
        )
        results = search_resp.points if hasattr(search_resp, "points") else []

        if not results:
            return {
                "status": "error",
                "error": "No relevant content found in the selected document(s)."
            }

        context = ""
        sources = set()
        for r in results:
            payload = r.payload or {}
            context += f"[From: {payload.get('source', 'unknown')}]\n{payload.get('content', '')}\n\n"
            sources.add(payload.get("source", "Unknown"))

        text_response = query_ollama(st.session_state.ollama_model, context, query)
        audio_path = generate_tts(text_response, st.session_state.tts_voice)

        return {
            "status": "success",
            "text_response": text_response,
            "audio_path": audio_path,
            "sources": list(sources)
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}


# ══════════════════════════════════════════════════════════════════
# SIDEBAR — unchanged model/voice/health logic
# ══════════════════════════════════════════════════════════════════

def sidebar_config():
    with st.sidebar:
        st.title("⚙️ Configuration")
        st.caption("100% local — no API keys needed")
        st.markdown("---")

        available_models = ["llama3.2", "mistral", "qwen2.5:7b"]
        try:
            fetched = ollama.list()
            if fetched and "models" in fetched:
                available_models = [m["model"] for m in fetched["models"]]
        except Exception:
            pass

        st.session_state.ollama_model = st.selectbox(
            "🤖 Ollama Model", options=available_models,
            help="Make sure Ollama is running: `ollama serve`"
        )

        voices = {
            "Jenny (US Female)": "en-US-JennyNeural",
            "Guy (US Male)": "en-US-GuyNeural",
            "Sonia (UK Female)": "en-GB-SoniaNeural",
            "Ryan (UK Male)": "en-GB-RyanNeural",
            "Natasha (AU Female)": "en-AU-NatashaNeural",
        }
        label = st.selectbox("🎤 TTS Voice", list(voices.keys()))
        st.session_state.tts_voice = voices[label]

        st.markdown("---")
        st.markdown("**Local services:**")

        qdrant_ok, ollama_ok = False, False
        try:
            QdrantClient(url="http://localhost:6333").get_collections()
            qdrant_ok = True
        except Exception:
            pass
        try:
            ollama.list()
            ollama_ok = True
        except Exception:
            pass

        st.markdown(f"{'🟢' if qdrant_ok else '🔴'} Qdrant `localhost:6333`")
        st.markdown(f"{'🟢' if ollama_ok else '🔴'} Ollama `localhost:11434`")
        if not qdrant_ok:
            st.warning("docker run -d -p 6333:6333 qdrant/qdrant")
        if not ollama_ok:
            st.warning("Run: ollama serve")

        # NEW — clear chat button in sidebar
        st.markdown("---")
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()


# ══════════════════════════════════════════════════════════════════
# NEW — Document Manager Panel
# Renders the full doc management UI:
#   1. Shows all docs in Qdrant (including previous sessions)
#   2. Lets user select which docs to query
#   3. File uploader to add new docs
# ══════════════════════════════════════════════════════════════════

def render_document_manager():
    st.markdown("### 📂 Document Manager")

    # Auto-connect Qdrant if not yet done (silent, no button needed)
    if st.session_state.client is None:
        try:
            client, emb_model = setup_qdrant()
            st.session_state.client = client
            st.session_state.embedding_model = emb_model
            st.session_state.setup_complete = True
        except Exception as e:
            st.error(f"Cannot connect to Qdrant: {e}")
            return

    client = st.session_state.client

    # ── Discover all docs already in Qdrant ──────────────────────
    all_sources = get_all_sources_from_qdrant(client)

    # ── File uploader ─────────────────────────────────────────────
    with st.expander("➕ Upload New Documents", expanded=not bool(all_sources)):
        uploaded_files = st.file_uploader(
            "Drag & drop files here",
            type=["txt", "md", "pdf", "html", "htm", "docx"],
            accept_multiple_files=True,
            key="file_uploader",
            help="Supported: .txt .md .pdf .html .docx"
        )
        if uploaded_files:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.caption(f"📎 {len(uploaded_files)} file(s) selected")
            with col2:
                process_btn = st.button("⚡ Process", type="primary", use_container_width=True)

            if process_btn:
                with st.status("Processing files...", expanded=True) as status:
                    try:
                        pages = []
                        for f in uploaded_files:
                            st.write(f"📄 Parsing `{f.name}`...")
                            content = parse_uploaded_file(f)
                            if content.strip():
                                pages.append({"content": content, "source": f.name})
                            else:
                                st.warning(f"⚠️ No text extracted from `{f.name}`")

                        st.write(f"💾 Embedding {len(pages)} doc(s)...")
                        counts = store_embeddings(client, st.session_state.embedding_model, pages)

                        # Update session registry
                        for src, cnt in counts.items():
                            st.session_state.doc_registry[src] = {
                                "chunks": cnt,
                                "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M")
                            }

                        status.update(label=f"✅ Stored {sum(counts.values())} chunks!", state="complete")
                        st.rerun()  # refresh doc list

                    except Exception as e:
                        status.update(label="❌ Error", state="error")
                        st.error(str(e))

    # ── Doc selector ──────────────────────────────────────────────
    if not all_sources:
        st.info("No documents loaded yet. Upload files above.")
        st.session_state.setup_complete = False
        return

    st.markdown(f"**{len(all_sources)} document(s) available** — select which to query:")

    # "Select all / none" quick toggles
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("✅ All", use_container_width=True):
            st.session_state.selected_sources = {s["source"] for s in all_sources}
            st.rerun()
    with col2:
        if st.button("⬜ None", use_container_width=True):
            st.session_state.selected_sources = set()
            st.rerun()

    # Individual checkboxes per document
    st.markdown("")
    newly_selected = set()
    for doc in all_sources:
        src = doc["source"]
        is_new = src in st.session_state.doc_registry  # uploaded this session
        label_suffix = " 🆕" if is_new else ""
        checked = src in st.session_state.selected_sources

        col_check, col_info = st.columns([1, 6])
        with col_check:
            selected = st.checkbox(
                label=" ",
                value=checked,
                key=f"doc_sel_{src}",
                label_visibility="collapsed"
            )
        with col_info:
            uploaded_label = (
                st.session_state.doc_registry[src]["uploaded_at"]
                if is_new
                else doc["uploaded_at"][:16].replace("T", " ")
            )
            st.markdown(
                f"**{src}**{label_suffix}  \n"
                f"<span style='color:gray;font-size:0.8em'>"
                f"{doc['chunk_count']} chunks · {uploaded_label}</span>",
                unsafe_allow_html=True
            )

        if selected:
            newly_selected.add(src)

    # Sync checkbox state back to session
    st.session_state.selected_sources = newly_selected

    if newly_selected:
        st.success(f"🔍 Querying: {', '.join(sorted(newly_selected))}")
        st.session_state.setup_complete = True
    else:
        st.warning("No documents selected — select at least one to ask questions.")
        st.session_state.setup_complete = False


# ══════════════════════════════════════════════════════════════════
# NEW — Chat History Renderer
# Renders all previous Q&A pairs stored in session_state.chat_history.
# Each entry has: question, text_response, audio_path, sources, timestamp
# ══════════════════════════════════════════════════════════════════

def render_chat_history():
    if not st.session_state.chat_history:
        return

    for i, entry in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.markdown(f"**{entry['question']}**")
            st.caption(entry.get("timestamp", ""))

        with st.chat_message("assistant"):
            st.write(entry["text_response"])

            audio_path = entry.get("audio_path")
            if audio_path and os.path.exists(audio_path):
                with st.expander("🔊 Play Audio"):
                    st.audio(audio_path, format="audio/mp3")
                    with open(audio_path, "rb") as f:
                        st.download_button(
                            "📥 Download",
                            f.read(),
                            file_name=f"response_{i+1}.mp3",
                            mime="audio/mp3",
                            key=f"dl_{i}"
                        )

            if entry.get("sources"):
                st.caption("📚 Sources: " + ", ".join(f"`{s}`" for s in entry["sources"]))


# ══════════════════════════════════════════════════════════════════
# MAIN
# CHANGED:
#   - Two-column layout: left = doc manager, right = chat
#   - Chat input uses st.chat_input (stays fixed at bottom)
#   - Results appended to chat_history, never replaced
# ══════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="Local Doc Voice Agent",
        page_icon="🦙",
        layout="wide"
    )
    init_session_state()
    sidebar_config()

    st.title("🦙 Local Documentation Voice Agent")
    st.caption("Drop your docs, ask questions, get voice answers — 100% offline.")

    left_col, right_col = st.columns([1, 2], gap="large")

    with left_col:
        render_document_manager()

    with right_col:
        st.markdown("### 💬 Chat")

        if not st.session_state.setup_complete:
            st.info("👈 Select at least one document from the left panel to start chatting.")
        else:
            render_chat_history()

            query = st.chat_input("Ask a question about the selected documents...")

            if query:
                with st.chat_message("user"):
                    st.markdown(f"**{query}**")

                with st.chat_message("assistant"):
                    with st.spinner(f"Searching with `{st.session_state.ollama_model}`..."):
                        result = process_query(query, st.session_state.selected_sources)

                    if result["status"] == "success":
                        st.write(result["text_response"])

                        if result.get("audio_path"):
                            with st.expander("🔊 Play Audio"):
                                st.audio(result["audio_path"], format="audio/mp3")
                                with open(result["audio_path"], "rb") as f:
                                    st.download_button(
                                        "📥 Download",
                                        f.read(),
                                        file_name="response.mp3",
                                        mime="audio/mp3",
                                        key=f"dl_new_{uuid.uuid4()}"
                                    )

                        if result.get("sources"):
                            st.caption("📚 Sources: " + ", ".join(f"`{s}`" for s in result["sources"]))

                        # Append to persistent chat history
                        st.session_state.chat_history.append({
                            "question": query,
                            "text_response": result["text_response"],
                            "audio_path": result.get("audio_path"),
                            "sources": result.get("sources", []),
                            "timestamp": datetime.now().strftime("%H:%M:%S"),
                            "docs_used": list(st.session_state.selected_sources)
                        })

                    else:
                        st.error(result.get("error", "Unknown error"))


if __name__ == "__main__":
    main()