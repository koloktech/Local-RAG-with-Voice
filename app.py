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
from openai import OpenAI

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

        # ── Company API (aivie-exchange) ──────────────────────────
        # llm_provider: "ollama" or "company_api"
        "llm_provider": "ollama",
        "company_api_key": "",
        "company_api_model": "si-gpt-oss-120b",
        "company_api_base_url": "https://aivie-exchange-tnt.sains.com.my/v1",
        "company_api_verified": False,

        # ── UI preferences ────────────────────────────────────────
        "font_size": 15,       # px, user-adjustable 12-20
        "settings_open": False,

        # ── Knowledge base panel ──────────────────────────────────
        "kb_expanded": True,       # collapsed/expanded state
        "doc_search": "",          # search filter string
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
# LLM QUERY — shared prompt builder + two provider functions
# query_ollama:       uses local Ollama (unchanged)
# query_company_api:  NEW — uses aivie-exchange OpenAI-compatible endpoint
# ══════════════════════════════════════════════════════════════════

def _build_prompt(context: str, question: str) -> str:
    """Shared prompt used by both providers."""
    return f"""You are a helpful documentation assistant.

Based on the following documentation content, answer the user's question clearly and concisely.
Write your response so it sounds natural when spoken aloud — avoid markdown formatting like bullet points or headers.
Use plain conversational sentences.

Documentation:
{context}

Question: {question}

Answer:"""


def query_ollama(model: str, context: str, question: str) -> str:
    """Local Ollama — unchanged."""
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": _build_prompt(context, question)}]
    )
    return response["message"]["content"]


def _get_base_url() -> str:
    """
    Normalise whatever the user pasted into a clean /v1 base URL.
    The OpenAI SDK always appends /chat/completions itself, so we must
    never pass a URL that already ends with that path.

    Handles all three common paste formats:
      https://host/v1                      -> https://host/v1
      https://host/v1/                     -> https://host/v1
      https://host/v1/chat/completions     -> https://host/v1

    Guards against None (st.text_input returns None before first render).
    """
    raw = st.session_state.get("company_api_base_url") or ""
    url = raw.strip().rstrip("/")
    for suffix in ("/chat/completions", "/chat"):
        if url.endswith(suffix):
            url = url[: -len(suffix)]
    return url


def _make_openai_client() -> OpenAI:
    """Build an OpenAI client pointed at the company gateway."""
    # Use .get() with fallback to guard against None from st.text_input
    model = (st.session_state.get("company_api_model") or "").strip()
    if not model:
        raise ValueError(
            "Model name is empty. Enter the model name in the sidebar (e.g. si-gpt-oss-120b)."
        )
    key = (st.session_state.get("company_api_key") or "").strip()
    if not key:
        raise ValueError("API key is empty. Enter your aivie-exchange API key in the sidebar.")
    base = _get_base_url()
    if not base:
        raise ValueError(
            "Base URL is empty. Enter the API URL in the sidebar "            "(e.g. https://aivie-exchange-tnt.sains.com.my/v1)."
        )
    return OpenAI(api_key=key, base_url=base)


def query_company_api(context: str, question: str) -> str:
    """Call aivie-exchange OpenAI-compatible endpoint."""
    client = _make_openai_client()
    response = client.chat.completions.create(
        model=st.session_state.company_api_model.strip(),
        messages=[{"role": "user", "content": _build_prompt(context, question)}],
    )
    return response.choices[0].message.content


def test_company_api() -> tuple[bool, str]:
    """Ping the API with a tiny request to verify key + model + URL."""
    try:
        client = _make_openai_client()
        response = client.chat.completions.create(
            model=st.session_state.company_api_model.strip(),
            messages=[{"role": "user", "content": "Reply with only the word: OK"}],
            max_tokens=5,
        )
        reply = response.choices[0].message.content.strip()
        return True, f"Connected! Model replied: {reply!r}"
    except ValueError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)


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

        # Route to the selected LLM provider
        if st.session_state.llm_provider == "company_api":
            text_response = query_company_api(context, query)
        else:
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
        st.markdown("---")

        # ── LLM Provider selector ─────────────────────────────────
        st.markdown("**🤖 LLM Provider**")
        provider = st.radio(
            "Provider",
            options=["ollama", "company_api"],
            format_func=lambda x: "🦙 Ollama (local)" if x == "ollama" else "🏢 Company API (aivie-exchange)",
            index=0 if st.session_state.llm_provider == "ollama" else 1,
            label_visibility="collapsed",
        )
        st.session_state.llm_provider = provider

        if provider == "ollama":
            # ── Ollama section (unchanged) ────────────────────────
            available_models = ["llama3.2", "mistral", "qwen2.5:7b"]
            try:
                fetched = ollama.list()
                if fetched and "models" in fetched:
                    available_models = [m["model"] for m in fetched["models"]]
            except Exception:
                pass
            st.session_state.ollama_model = st.selectbox(
                "🤖 Ollama Model", options=available_models,
                help="Make sure Ollama is running: ollama serve"
            )

        else:
            # ── Company API section (NEW) ─────────────────────────
            st.session_state.company_api_base_url = st.text_input(
                "Base URL",
                value=st.session_state.company_api_base_url,
                placeholder="https://aivie-exchange-tnt.sains.com.my/v1",
                help="Paste /v1 base URL or full /v1/chat/completions — both work"
            )
            st.session_state.company_api_key = st.text_input(
                "API Key",
                value=st.session_state.company_api_key,
                type="password",
                placeholder="sk-xxxx...",
            )
            st.session_state.company_api_model = st.text_input(
                "Model name",
                value=st.session_state.company_api_model,
                placeholder="si-gpt-oss-120b",
                help="Exact model name as provided by aivie-exchange"
            )

            # Test connection button
            col_test, col_status = st.columns([1, 2])
            with col_test:
                test_btn = st.button("🔌 Test", use_container_width=True)
            if test_btn:
                if not st.session_state.company_api_key or not st.session_state.company_api_base_url:
                    st.warning("Fill in URL and API key first.")
                else:
                    with st.spinner("Testing..."):
                        ok, msg = test_company_api()
                    st.session_state.company_api_verified = ok
                    if ok:
                        st.success(msg)
                    else:
                        st.error(f"Failed: {msg}")

            # Show persistent status badge
            if st.session_state.company_api_verified:
                st.markdown("🟢 API verified this session")
            elif st.session_state.company_api_key:
                st.markdown("🟡 Not yet tested — click Test")

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
        st.markdown("**🔌 Service Status**")

        qdrant_ok = False
        try:
            QdrantClient(url="http://localhost:6333").get_collections()
            qdrant_ok = True
        except Exception:
            pass

        st.markdown(f"{'🟢' if qdrant_ok else '🔴'} Qdrant `localhost:6333`")
        if not qdrant_ok:
            st.warning("docker run -d -p 6333:6333 qdrant/qdrant")

        if provider == "ollama":
            ollama_ok = False
            try:
                ollama.list()
                ollama_ok = True
            except Exception:
                pass
            st.markdown(f"{'🟢' if ollama_ok else '🔴'} Ollama `localhost:11434`")
            if not ollama_ok:
                st.warning("Run: ollama serve")
        else:
            api_status = "🟢 API verified" if st.session_state.company_api_verified else "🟡 API not tested"
            st.markdown(api_status)

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
    # ── Collapsible header button ─────────────────────────────────
    # Shows a 📁 button with doc count badge; clicking toggles panel
    n_selected = len(st.session_state.selected_sources)
    n_total    = 0  # will be computed below if expanded

    kb_open = st.session_state.kb_expanded

    # Header row: icon-button + selection summary
    h1, h2 = st.columns([6, 1])
    with h1:
        sel_badge = (
            f'<span class="badge badge-green" style="margin-left:0.5rem">{n_selected} selected</span>'
            if n_selected > 0
            else '<span class="badge badge-amber" style="margin-left:0.5rem">none selected</span>'
        )
        st.markdown(
            f"""<div class="kb-header" style="margin-bottom:0.4rem">
                <div style="display:flex;align-items:center;gap:0.5rem">
                    <div class="kb-title">📁 Knowledge Base</div>
                    {sel_badge}
                </div>
            </div>""",
            unsafe_allow_html=True
        )
    with h2:
        toggle_label = "▲ Hide" if kb_open else "▼ Show"
        if st.button(toggle_label, key="kb_toggle", use_container_width=True):
            st.session_state.kb_expanded = not kb_open
            st.rerun()

    # ── Collapsed state: just show selected file names compactly ──
    if not kb_open:
        if n_selected:
            chips = "".join(
                f"<span class='source-chip'>{s}</span>"
                for s in sorted(st.session_state.selected_sources)
            )
            st.markdown(
                f"<div style='margin-top:0.3rem;line-height:2'>{chips}</div>",
                unsafe_allow_html=True
            )
        else:
            st.caption("No documents selected. Expand to choose.")
        return  # ← panel is collapsed, stop here

    # ══ EXPANDED PANEL ════════════════════════════════════════════

    # Auto-connect Qdrant silently
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
    all_sources = get_all_sources_from_qdrant(client)

    # ── File uploader expander ─────────────────────────────────────
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
                            content_text = parse_uploaded_file(f)
                            if content_text.strip():
                                pages.append({"content": content_text, "source": f.name})
                            else:
                                st.warning(f"⚠️ No text extracted from `{f.name}`")

                        st.write(f"💾 Embedding {len(pages)} doc(s)...")
                        counts = store_embeddings(client, st.session_state.embedding_model, pages)

                        for src, cnt in counts.items():
                            st.session_state.doc_registry[src] = {
                                "chunks": cnt,
                                "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M")
                            }

                        status.update(label=f"✅ Stored {sum(counts.values())} chunks!", state="complete")
                        st.rerun()

                    except Exception as e:
                        status.update(label="❌ Error", state="error")
                        st.error(str(e))

    # ── No docs yet ───────────────────────────────────────────────
    if not all_sources:
        st.info("No documents loaded yet. Upload files above.")
        st.session_state.setup_complete = False
        return

    # ── Search bar ────────────────────────────────────────────────
    st.markdown("")
    search_val = st.text_input(
        "search_docs",
        value=st.session_state.doc_search,
        placeholder="🔍  Search documents...",
        label_visibility="collapsed",
        key="doc_search_input",
    )
    # Sync search to session state
    if search_val != st.session_state.doc_search:
        st.session_state.doc_search = search_val
        st.rerun()

    # Filter sources by search term (case-insensitive)
    query_term = st.session_state.doc_search.strip().lower()
    filtered_sources = (
        [s for s in all_sources if query_term in s["source"].lower()]
        if query_term
        else all_sources
    )

    # ── Select all / none row ─────────────────────────────────────
    total_shown = len(filtered_sources)
    count_label = (
        f"{total_shown} of {len(all_sources)} document(s)"
        if query_term
        else f"{len(all_sources)} document(s) available"
    )
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("✅ All", use_container_width=True, key="sel_all"):
            # Select all FILTERED docs (respects search)
            st.session_state.selected_sources = {s["source"] for s in filtered_sources}
            st.rerun()
    with col2:
        if st.button("⬜ None", use_container_width=True, key="sel_none"):
            st.session_state.selected_sources = set()
            st.rerun()
    with col3:
        st.markdown(
            f"<div style='padding-top:0.45rem;color:var(--text-muted);font-size:var(--fs-xs)'>{count_label}</div>",
            unsafe_allow_html=True
        )

    # ── No results for search ─────────────────────────────────────
    if not filtered_sources:
        st.caption(f'No documents match "{st.session_state.doc_search}".')
        st.session_state.setup_complete = False
        return

    # ── Document checkboxes ───────────────────────────────────────
    st.markdown("")
    newly_selected = set(st.session_state.selected_sources)  # start with current selection

    for doc in filtered_sources:
        src = doc["source"]
        is_new = src in st.session_state.doc_registry
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
            badge_cls = "badge-blue" if is_new else "badge-green"
            badge_txt = "New" if is_new else "Indexed"
            st.markdown(
                f"""<div style="display:flex;align-items:center;justify-content:space-between;padding:0.1rem 0">
                    <div>
                        <div class="file-row-name">{src}</div>
                        <div class="file-row-meta">{doc["chunk_count"]} chunks &middot; {uploaded_label}</div>
                    </div>
                    <span class="badge {badge_cls}">{badge_txt}</span>
                </div>""",
                unsafe_allow_html=True
            )

        if selected:
            newly_selected.add(src)
        else:
            newly_selected.discard(src)

    st.session_state.selected_sources = newly_selected

    if newly_selected:
        st.success(f"🔍 {len(newly_selected)} doc(s) active for RAG")
        st.session_state.setup_complete = True
    else:
        st.warning("Select at least one document to start chatting.")
        st.session_state.setup_complete = False

# ══════════════════════════════════════════════════════════════════
# NEW — Chat History Renderer
# Renders all previous Q&A pairs stored in session_state.chat_history.
# Each entry has: question, text_response, audio_path, sources, timestamp
# ══════════════════════════════════════════════════════════════════

def build_chat_export(chat_history: list) -> str:
    """Serialise chat history to a clean plain-text string for download."""
    lines = []
    lines.append("=" * 60)
    lines.append("  Knowledge Voice Agent — Chat Export")
    lines.append(f"  Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 60)
    lines.append("")

    for i, entry in enumerate(chat_history, 1):
        ts        = entry.get("timestamp", "")
        docs_used = ", ".join(entry.get("docs_used", [])) or "—"

        lines.append(f"[{i}]  {ts}")
        lines.append(f"Docs : {docs_used}")
        lines.append(f"You  : {entry['question']}")
        lines.append(f"AI   : {entry['text_response']}")
        if entry.get("sources"):
            lines.append(f"Src  : {', '.join(entry['sources'])}")
        lines.append("-" * 60)
        lines.append("")

    return "\n".join(lines)


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
                chips = "".join(f"<span class='source-chip'>{s}</span>" for s in entry["sources"])
                st.markdown(f"<div style='margin-top:0.4rem'>📚 {chips}</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# MAIN
# CHANGED:
#   - Two-column layout: left = doc manager, right = chat
#   - Chat input uses st.chat_input (stays fixed at bottom)
#   - Results appended to chat_history, never replaced
# ══════════════════════════════════════════════════════════════════

def inject_css():
    fs = st.session_state.get("font_size", 15)
    fs_sm  = round(fs * 0.85, 1)   # ~small labels
    fs_xs  = round(fs * 0.75, 1)   # ~captions / meta
    fs_lg  = round(fs * 1.15, 1)   # ~titles
    fs_xl  = round(fs * 1.35, 1)   # ~page title

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {{
        --bg-base:      #0e0e12;
        --bg-panel:     #16161d;
        --bg-card:      #1e1e28;
        --bg-hover:     #252532;
        --bg-input:     #1a1a24;
        --border:       #2a2a38;
        --border-light: #33334a;
        --accent:       #4f6ef7;
        --accent-dim:   #2d3f8f;
        --accent-glow:  rgba(79,110,247,0.18);
        --green:        #22c55e;
        --green-dim:    rgba(34,197,94,0.15);
        --amber:        #f59e0b;
        --red:          #ef4444;
        --text-primary: #e8e8f0;
        --text-secondary:#9898b0;
        --text-muted:   #5a5a72;
        --radius:       12px;
        --radius-sm:    8px;
        --fs-base:      {fs}px;
        --fs-sm:        {fs_sm}px;
        --fs-xs:        {fs_xs}px;
        --fs-lg:        {fs_lg}px;
        --fs-xl:        {fs_xl}px;
    }}

    html, body, .stApp {{
        background-color: var(--bg-base) !important;
        font-family: 'DM Sans', sans-serif !important;
        color: var(--text-primary) !important;
        font-size: var(--fs-base) !important;
    }}

    #MainMenu, footer, header {{ visibility: hidden; }}
    .stDeployButton {{ display: none; }}

    .main .block-container {{
        padding: 1.5rem 2rem !important;
        max-width: 100% !important;
    }}

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {{
        background: var(--bg-panel) !important;
        border-right: 1px solid var(--border) !important;
    }}
    [data-testid="stSidebar"] > div:first-child {{
        padding: 1.5rem 1.2rem !important;
    }}
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stRadio label {{
        color: var(--text-secondary) !important;
        font-size: var(--fs-sm) !important;
        letter-spacing: 0.02em;
    }}
    [data-testid="stSidebar"] h1 {{
        font-size: var(--fs-lg) !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
    }}
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] .stTextInput input {{
        background: var(--bg-input) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--text-primary) !important;
        font-size: var(--fs-sm) !important;
    }}
    [data-testid="stSidebar"] input:focus {{
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px var(--accent-glow) !important;
    }}
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {{
        background: var(--bg-input) !important;
        border-color: var(--border) !important;
        border-radius: var(--radius-sm) !important;
    }}

    /* ── Typography scale ── */
    h1 {{
        font-size: var(--fs-xl) !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        letter-spacing: -0.02em !important;
        margin-bottom: 0.1rem !important;
    }}
    h2 {{ font-size: var(--fs-lg) !important; color: var(--text-primary) !important; }}
    h3 {{
        font-size: var(--fs-xs) !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        color: var(--text-muted) !important;
        margin-bottom: 0.8rem !important;
        margin-top: 0.2rem !important;
    }}
    p, li, span, div {{
        font-size: var(--fs-base) !important;
    }}

    /* ── Cards / expanders ── */
    [data-testid="stExpander"] {{
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        margin-bottom: 0.6rem !important;
    }}
    [data-testid="stExpander"] summary {{
        color: var(--text-secondary) !important;
        font-size: var(--fs-base) !important;
        font-weight: 500 !important;
        padding: 0.75rem 1rem !important;
    }}
    [data-testid="stExpander"] summary:hover {{
        color: var(--text-primary) !important;
        background: var(--bg-hover) !important;
    }}

    /* ── File uploader ── */
    [data-testid="stFileUploader"] {{
        background: var(--bg-card) !important;
        border: 1.5px dashed var(--border-light) !important;
        border-radius: var(--radius) !important;
        padding: 1.2rem !important;
        transition: border-color 0.2s, background 0.2s;
    }}
    [data-testid="stFileUploader"]:hover {{
        border-color: var(--accent) !important;
        background: var(--accent-glow) !important;
    }}
    [data-testid="stFileUploader"] label {{
        color: var(--text-secondary) !important;
        font-size: var(--fs-base) !important;
    }}

    /* ── Buttons ── */
    .stButton > button[kind="primary"] {{
        background: var(--accent) !important;
        color: #fff !important;
        border: none !important;
        border-radius: var(--radius-sm) !important;
        font-size: var(--fs-sm) !important;
        font-weight: 500 !important;
        padding: 0.45rem 1rem !important;
        transition: opacity 0.15s, box-shadow 0.15s;
    }}
    .stButton > button[kind="primary"]:hover {{
        opacity: 0.88 !important;
        box-shadow: 0 4px 14px var(--accent-glow) !important;
    }}
    .stButton > button:not([kind="primary"]) {{
        background: var(--bg-card) !important;
        color: var(--text-secondary) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-sm) !important;
        font-size: var(--fs-sm) !important;
        font-weight: 500 !important;
        transition: background 0.15s, color 0.15s;
    }}
    .stButton > button:not([kind="primary"]):hover {{
        background: var(--bg-hover) !important;
        color: var(--text-primary) !important;
        border-color: var(--border-light) !important;
    }}

    /* ── Alerts ── */
    [data-testid="stAlert"] {{
        border-radius: var(--radius-sm) !important;
        border: none !important;
        font-size: var(--fs-sm) !important;
        padding: 0.6rem 0.9rem !important;
    }}

    /* ── Chat messages ── */
    [data-testid="stChatMessage"] {{
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        padding: 0.9rem 1.1rem !important;
        margin-bottom: 0.6rem !important;
    }}
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {{
        border-left: 3px solid var(--accent) !important;
        background: #181824 !important;
    }}
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {{
        border-left: 3px solid var(--green) !important;
    }}
    [data-testid="stChatMessage"] p {{
        color: var(--text-primary) !important;
        font-size: var(--fs-base) !important;
        line-height: 1.65 !important;
    }}

    /* ── Chat input ── */
    [data-testid="stChatInput"] {{
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
    }}
    [data-testid="stChatInput"]:focus-within {{
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px var(--accent-glow) !important;
    }}
    [data-testid="stChatInput"] textarea {{
        background: transparent !important;
        color: var(--text-primary) !important;
        font-size: var(--fs-base) !important;
        font-family: 'DM Sans', sans-serif !important;
    }}
    [data-testid="stChatInput"] textarea::placeholder {{
        color: var(--text-muted) !important;
    }}
    [data-testid="stChatInput"] button {{
        background: var(--accent) !important;
        border-radius: 8px !important;
        color: white !important;
    }}

    /* ── All inputs / selects ── */
    input, textarea, select {{
        font-size: var(--fs-base) !important;
        color: var(--text-primary) !important;
    }}

    /* ── Dividers ── */
    hr {{ border-color: var(--border) !important; margin: 1rem 0 !important; }}

    /* ── Captions ── */
    .stCaption, small, [data-testid="stCaptionContainer"] {{
        color: var(--text-muted) !important;
        font-size: var(--fs-xs) !important;
    }}

    /* ── Scrollbar ── */
    ::-webkit-scrollbar {{ width: 4px; height: 4px; }}
    ::-webkit-scrollbar-track {{ background: var(--bg-base); }}
    ::-webkit-scrollbar-thumb {{ background: var(--border-light); border-radius: 99px; }}

    /* ── Radio ── */
    [data-testid="stRadio"] label {{
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-sm) !important;
        padding: 0.4rem 0.75rem !important;
        margin-bottom: 0.3rem !important;
        font-size: var(--fs-sm) !important;
        cursor: pointer;
        transition: background 0.15s;
    }}
    [data-testid="stRadio"] label:hover {{
        background: var(--bg-hover) !important;
    }}

    /* ── Checkboxes ── */
    [data-testid="stCheckbox"] label {{
        color: var(--text-secondary) !important;
        font-size: var(--fs-base) !important;
    }}

    /* ── Columns gap ── */
    [data-testid="column"] {{ padding: 0 0.6rem !important; }}

    /* ── Slider ── */
    [data-testid="stSlider"] label {{
        color: var(--text-secondary) !important;
        font-size: var(--fs-sm) !important;
    }}
    [data-testid="stSlider"] [data-testid="stTickBar"] span {{
        color: var(--text-muted) !important;
        font-size: var(--fs-xs) !important;
    }}

    /* ── Custom component classes ── */
    .app-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.5rem 0 1.2rem 0;
        border-bottom: 1px solid var(--border);
        margin-bottom: 1.2rem;
    }}
    .app-header-title {{
        font-size: var(--fs-lg) !important;
        font-weight: 600;
        color: var(--text-primary);
        letter-spacing: -0.02em;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}
    .app-header-sub {{
        font-size: var(--fs-xs) !important;
        color: var(--text-muted);
        margin-top: 0.15rem;
    }}
    .badge {{
        display: inline-block;
        padding: 0.18rem 0.55rem;
        border-radius: 99px;
        font-size: var(--fs-xs) !important;
        font-weight: 600;
        letter-spacing: 0.04em;
    }}
    .badge-green  {{ background: var(--green-dim);  color: var(--green); }}
    .badge-amber  {{ background: rgba(245,158,11,.15); color: var(--amber); }}
    .badge-blue   {{ background: var(--accent-glow); color: var(--accent); }}
    .badge-red    {{ background: rgba(239,68,68,.15); color: var(--red); }}

    .kb-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.8rem;
    }}
    .kb-title {{
        font-size: var(--fs-xs) !important;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-muted);
    }}

    .file-row-name {{
        font-size: var(--fs-base) !important;
        color: var(--text-primary);
        font-weight: 500;
        font-family: 'JetBrains Mono', monospace;
        word-break: break-all;
    }}
    .file-row-meta {{
        font-size: var(--fs-xs) !important;
        color: var(--text-muted);
        margin-top: 0.1rem;
    }}

    .chat-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding-bottom: 0.9rem;
        border-bottom: 1px solid var(--border);
        margin-bottom: 1rem;
    }}
    .chat-header-left {{
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }}
    .chat-avatar {{
        width: 36px; height: 36px;
        background: var(--accent);
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 1rem;
        flex-shrink: 0;
    }}
    .chat-title {{
        font-size: var(--fs-base) !important;
        font-weight: 600;
        color: var(--text-primary);
    }}
    .chat-subtitle {{
        font-size: var(--fs-xs) !important;
        color: var(--green);
        display: flex; align-items: center; gap: 0.3rem;
    }}
    .chat-subtitle::before {{
        content: '';
        width: 6px; height: 6px;
        background: var(--green);
        border-radius: 50%;
        display: inline-block;
    }}

    .source-chip {{
        display: inline-block;
        background: var(--bg-hover);
        border: 1px solid var(--border);
        border-radius: 99px;
        padding: 0.15rem 0.6rem;
        font-size: var(--fs-xs) !important;
        color: var(--accent);
        font-family: 'JetBrains Mono', monospace;
        margin-right: 0.3rem;
        margin-bottom: 0.2rem;
    }}

    .disclaimer {{
        text-align: center;
        font-size: var(--fs-xs) !important;
        color: var(--text-muted);
        padding-top: 0.5rem;
        border-top: 1px solid var(--border);
        margin-top: 0.5rem;
    }}

    /* ── Settings panel overlay ── */
    .settings-panel {{
        background: var(--bg-panel);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.5rem;
    }}
    .settings-panel h4 {{
        font-size: var(--fs-xs) !important;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-muted);
        margin-bottom: 0.9rem;
        margin-top: 0;
    }}
    .settings-section {{
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius-sm);
        padding: 1rem 1.1rem;
        margin-bottom: 0.8rem;
    }}
    </style>
    """, unsafe_allow_html=True)


def render_settings_panel():
    """In-page settings panel — shown when user clicks ⚙️ Settings."""

    st.markdown('<div class="settings-panel">', unsafe_allow_html=True)

    # ── Header row with close button ─────────────────────────────
    col_title, col_close = st.columns([5, 1])
    with col_title:
        st.markdown('<h4 style="margin:0;padding:0.3rem 0 0.8rem 0;font-size:0.85rem;font-weight:700;color:#9898b0;text-transform:uppercase;letter-spacing:0.08em">⚙️ Settings</h4>', unsafe_allow_html=True)
    with col_close:
        if st.button("✕ Close", key="close_settings", use_container_width=True):
            st.session_state.settings_open = False
            st.rerun()

    # ── Section 1: LLM Provider ───────────────────────────────────
    st.markdown('<div class="settings-section">', unsafe_allow_html=True)
    st.markdown("**🤖 LLM Provider**")
    provider = st.radio(
        "Provider",
        options=["ollama", "company_api"],
        format_func=lambda x: "🦙 Ollama (local)" if x == "ollama" else "🏢 Company API (aivie-exchange)",
        index=0 if st.session_state.llm_provider == "ollama" else 1,
        horizontal=True,
        key="settings_provider",
        label_visibility="collapsed",
    )
    st.session_state.llm_provider = provider

    if provider == "ollama":
        available_models = ["llama3.2", "mistral", "qwen2.5:7b"]
        try:
            fetched = ollama.list()
            if fetched and "models" in fetched:
                available_models = [m["model"] for m in fetched["models"]]
        except Exception:
            pass
        st.session_state.ollama_model = st.selectbox(
            "Ollama Model", options=available_models,
            key="settings_ollama_model",
            help="Run: ollama serve"
        )
    else:
        c1, c2 = st.columns(2)
        with c1:
            val = st.text_input(
                "Base URL",
                value=st.session_state.company_api_base_url or "",
                placeholder="https://aivie-exchange-tnt.sains.com.my/v1",
                key="settings_base_url",
            )
            if val is not None:
                st.session_state.company_api_base_url = val
        with c2:
            val = st.text_input(
                "Model name",
                value=st.session_state.company_api_model or "",
                placeholder="si-gpt-oss-120b",
                key="settings_model_name",
            )
            if val is not None:
                st.session_state.company_api_model = val

        val = st.text_input(
            "API Key",
            value=st.session_state.company_api_key or "",
            type="password",
            placeholder="sk-xxxx...",
            key="settings_api_key",
        )
        if val is not None:
            st.session_state.company_api_key = val

        col_test, col_status = st.columns([1, 3])
        with col_test:
            if st.button("🔌 Test Connection", key="settings_test_btn", use_container_width=True):
                with st.spinner("Testing..."):
                    ok, msg = test_company_api()
                st.session_state.company_api_verified = ok
                if ok:
                    st.success(msg)
                else:
                    st.error(f"Failed: {msg}")
        with col_status:
            if st.session_state.company_api_verified:
                st.markdown('<span class="badge badge-green">🟢 Verified</span>', unsafe_allow_html=True)
            elif st.session_state.company_api_key:
                st.markdown('<span class="badge badge-amber">🟡 Not tested</span>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Section 2: TTS Voice ──────────────────────────────────────
    st.markdown('<div class="settings-section">', unsafe_allow_html=True)
    st.markdown("**🎤 TTS Voice**")
    voices = {
        "Jenny (US Female)": "en-US-JennyNeural",
        "Guy (US Male)": "en-US-GuyNeural",
        "Sonia (UK Female)": "en-GB-SoniaNeural",
        "Ryan (UK Male)": "en-GB-RyanNeural",
        "Natasha (AU Female)": "en-AU-NatashaNeural",
    }
    label = st.selectbox("Voice", list(voices.keys()), key="settings_voice", label_visibility="collapsed")
    st.session_state.tts_voice = voices[label]
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Section 3: Font Size ──────────────────────────────────────
    st.markdown('<div class="settings-section">', unsafe_allow_html=True)
    st.markdown("**🔠 Font Size**")
    c1, c2 = st.columns([3, 1])
    with c1:
        new_fs = st.slider(
            "Font size (px)",
            min_value=12, max_value=20,
            value=st.session_state.font_size,
            step=1,
            key="settings_font_slider",
            label_visibility="collapsed",
        )
    with c2:
        st.markdown(f'<div style="padding-top:0.6rem;font-weight:600;color:var(--text-primary)">{new_fs}px</div>', unsafe_allow_html=True)
    if new_fs != st.session_state.font_size:
        st.session_state.font_size = new_fs
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Section 4: Service status ─────────────────────────────────
    st.markdown('<div class="settings-section">', unsafe_allow_html=True)
    st.markdown("**🔌 Service Status**")
    sc1, sc2 = st.columns(2)
    with sc1:
        qdrant_ok = False
        try:
            QdrantClient(url="http://localhost:6333").get_collections()
            qdrant_ok = True
        except Exception:
            pass
        st.markdown(f"{'🟢' if qdrant_ok else '🔴'} **Qdrant** `localhost:6333`")
        if not qdrant_ok:
            st.caption("docker run -d -p 6333:6333 qdrant/qdrant")
    with sc2:
        if provider == "ollama":
            ollama_ok = False
            try:
                ollama.list()
                ollama_ok = True
            except Exception:
                pass
            st.markdown(f"{'🟢' if ollama_ok else '🔴'} **Ollama** `localhost:11434`")
            if not ollama_ok:
                st.caption("Run: ollama serve")
        else:
            verified = st.session_state.company_api_verified
            st.markdown(f"{'🟢' if verified else '🟡'} **aivie-exchange API**")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Section 5: Chat ───────────────────────────────────────────
    st.markdown('<div class="settings-section">', unsafe_allow_html=True)
    st.markdown("**💬 Chat**")
    if st.button("🗑️ Clear Chat History", key="settings_clear_chat", use_container_width=False):
        st.session_state.chat_history = []
        st.session_state.settings_open = False
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # close settings-panel


def main():
    st.set_page_config(
        page_title="Knowledge Voice Agent",
        page_icon="🎙️",
        layout="wide",
        initial_sidebar_state="collapsed"   # sidebar hidden — settings now in-page
    )
    init_session_state()
    inject_css()

    # ── Header ────────────────────────────────────────────────────
    rag_status = "RAG Active" if st.session_state.setup_complete else "No docs selected"
    badge_cls  = "badge-green" if st.session_state.setup_complete else "badge-amber"
    provider_name = (
        st.session_state.get("company_api_model", "API")
        if st.session_state.get("llm_provider") == "company_api"
        else st.session_state.get("ollama_model", "Ollama")
    )

    hcol1, hcol2 = st.columns([5, 1])
    with hcol1:
        st.markdown(f"""
        <div class="app-header">
            <div>
                <div class="app-header-title">🎙️ Knowledge Voice Agent</div>
                <div class="app-header-sub">Powered by {provider_name} · edge-tts · Qdrant</div>
            </div>
            <div style="margin-left:1rem">
                <span class="badge {badge_cls}">{rag_status}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with hcol2:
        if st.button(
            "⚙️ Settings" if not st.session_state.settings_open else "✕ Close Settings",
            key="toggle_settings",
            use_container_width=True,
        ):
            st.session_state.settings_open = not st.session_state.settings_open
            st.rerun()

    # ── Settings panel (shown inline when toggled) ────────────────
    if st.session_state.settings_open:
        render_settings_panel()
        st.markdown("---")

    left_col, right_col = st.columns([1, 2], gap="large")

    with left_col:
        render_document_manager()

    with right_col:
        # Chat panel header
        chat_status = "RAG Active" if st.session_state.setup_complete else "Idle"

        ch1, ch2 = st.columns([4, 1])
        with ch1:
            st.markdown(f"""
            <div class="chat-header">
                <div class="chat-header-left">
                    <div class="chat-avatar">🎙️</div>
                    <div>
                        <div class="chat-title">AI Assistant</div>
                        <div class="chat-subtitle">{chat_status}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with ch2:
            # Export button — only shown when there is chat history
            if st.session_state.chat_history:
                export_txt = build_chat_export(st.session_state.chat_history)
                export_filename = f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                st.download_button(
                    label="💾 Export",
                    data=export_txt.encode("utf-8"),
                    file_name=export_filename,
                    mime="text/plain",
                    key="export_chat_btn",
                    use_container_width=True,
                    help="Download full chat history as a .txt file",
                )

        if not st.session_state.setup_complete:
            st.markdown("""
            <div style="background:var(--bg-card);border:1px solid var(--border);border-radius:var(--radius);
                        padding:1.2rem 1.4rem;color:var(--text-secondary);font-size:0.85rem;">
                👋 Hello! Select one or more documents from the <strong>Knowledge Base</strong> panel on the left
                to activate RAG, then ask me anything about them.
            </div>
            """, unsafe_allow_html=True)
        else:
            render_chat_history()

            query = st.chat_input("Ask a question about your documents...")

            if query:
                with st.chat_message("user"):
                    st.markdown(f"**{query}**")

                with st.chat_message("assistant"):
                    provider_label = (
                        f"`{st.session_state.company_api_model}` via aivie-exchange"
                        if st.session_state.llm_provider == "company_api"
                        else f"`{st.session_state.ollama_model}` via Ollama"
                    )
                    with st.spinner(f"Searching with {provider_label}..."):
                        result = process_query(query, st.session_state.selected_sources)

                    if result["status"] == "success":
                        st.write(result["text_response"])

                        if result.get("sources"):
                            chips = "".join(f"<span class='source-chip'>{s}</span>" for s in result["sources"])
                            st.markdown(f"<div style='margin-top:0.5rem'>📚 {chips}</div>", unsafe_allow_html=True)

                        if result.get("audio_path"):
                            with st.expander("🔊 Play Voice Response"):
                                st.audio(result["audio_path"], format="audio/mp3")
                                with open(result["audio_path"], "rb") as f:
                                    st.download_button(
                                        "📥 Download",
                                        f.read(),
                                        file_name="response.mp3",
                                        mime="audio/mp3",
                                        key=f"dl_new_{uuid.uuid4()}"
                                    )

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

        st.markdown("<div class='disclaimer'>AI can make mistakes. Verify important information.</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()