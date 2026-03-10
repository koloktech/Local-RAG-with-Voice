import streamlit as st
import asyncio
import uuid
import tempfile
import os
from datetime import datetime
from typing import List, Dict

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from fastembed import TextEmbedding
import ollama
import edge_tts

# ── Windows asyncio fix ───────────────────────────────────────────
import sys
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ── Session State ─────────────────────────────────────────────────

def init_session_state():
    defaults = {
        "setup_complete": False,
        "client": None,
        "embedding_model": None,
        "ollama_model": "llama3.2",
        "tts_voice": "en-US-JennyNeural",
        "docs_loaded": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ── Qdrant Setup ──────────────────────────────────────────────────

def setup_qdrant(collection_name="docs_embeddings"):
    client = QdrantClient(url="http://localhost:6333")
    embedding_model = TextEmbedding()
    test_emb = list(embedding_model.embed(["test"]))[0]
    dim = len(test_emb)

    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )
    except Exception as e:
        if "already exists" not in str(e):
            raise e

    return client, embedding_model

# ── File Parsing ──────────────────────────────────────────────────

def parse_uploaded_file(uploaded_file) -> str:
    """Extract text from uploaded file based on its type."""
    filename = uploaded_file.name.lower()
    raw = uploaded_file.read()

    if filename.endswith(".txt") or filename.endswith(".md"):
        return raw.decode("utf-8", errors="ignore")

    elif filename.endswith(".pdf"):
        try:
            import pypdf
            import io
            reader = pypdf.PdfReader(io.BytesIO(raw))
            return "\n\n".join(
                page.extract_text() for page in reader.pages if page.extract_text()
            )
        except ImportError:
            st.error("Install pypdf for PDF support: pip install pypdf")
            return ""

    elif filename.endswith(".html") or filename.endswith(".htm"):
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(raw, "html.parser")
            # Remove scripts and styles
            for tag in soup(["script", "style", "nav", "footer"]):
                tag.decompose()
            return soup.get_text(separator="\n", strip=True)
        except ImportError:
            # Fallback: strip tags manually
            import re
            text = raw.decode("utf-8", errors="ignore")
            return re.sub(r"<[^>]+>", " ", text)

    elif filename.endswith(".docx"):
        try:
            import docx
            import io
            doc = docx.Document(io.BytesIO(raw))
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except ImportError:
            st.error("Install python-docx for Word support: pip install python-docx")
            return ""

    else:
        # Try decoding as plain text
        return raw.decode("utf-8", errors="ignore")


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """Split long text into overlapping chunks for better retrieval."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c for c in chunks if c.strip()]


# ── Store Embeddings ──────────────────────────────────────────────

def store_embeddings(client, embedding_model, pages: List[Dict], collection_name="docs_embeddings"):
    stored = 0
    for page in pages:
        content = page.get("content", "")
        if not content.strip():
            continue

        # Chunk large documents
        chunks = chunk_text(content)
        for i, chunk in enumerate(chunks):
            embedding = list(embedding_model.embed([chunk]))[0]
            client.upsert(
                collection_name=collection_name,
                points=[models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding.tolist(),
                    payload={
                        "content": chunk,
                        "source": page.get("source", "unknown"),
                        "chunk_index": i,
                        "crawl_date": datetime.now().isoformat()
                    }
                )]
            )
            stored += 1
    return stored

# ── Query with Ollama ─────────────────────────────────────────────

def query_ollama(model: str, context: str, question: str) -> str:
    prompt = f"""You are a helpful documentation assistant.

Based on the following documentation content, answer the user's question clearly and concisely.
Write your response so it sounds natural when spoken aloud — avoid markdown formatting like bullet points or headers.
Use plain conversational sentences.

Documentation:
{context}

Question: {question}

Answer:"""

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]

# ── TTS with edge-tts ─────────────────────────────────────────────

async def _generate_tts_async(text: str, voice: str) -> str:
    temp_path = os.path.join(tempfile.gettempdir(), f"tts_{uuid.uuid4()}.mp3")
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(temp_path)
    return temp_path

def generate_tts(text: str, voice: str) -> str:
    """Sync wrapper for TTS that handles Windows event loop correctly."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_generate_tts_async(text, voice))
    finally:
        loop.close()

# ── Full Query Pipeline ───────────────────────────────────────────

def process_query(query: str) -> Dict:
    try:
        client = st.session_state.client
        embedding_model = st.session_state.embedding_model

        # 1. Embed query and search Qdrant
        query_emb = list(embedding_model.embed([query]))[0]
        search_resp = client.query_points(
            collection_name="docs_embeddings",
            query=query_emb.tolist(),
            limit=4,
            with_payload=True
        )
        results = search_resp.points if hasattr(search_resp, "points") else []

        if not results:
            return {
                "status": "error",
                "error": "No relevant content found. Make sure you've uploaded and processed your files."
            }

        # 2. Build context from top results
        context = ""
        sources = set()
        for r in results:
            payload = r.payload or {}
            context += f"[From: {payload.get('source', 'unknown')}]\n{payload.get('content', '')}\n\n"
            sources.add(payload.get("source", "Unknown"))

        # 3. Ask Ollama
        text_response = query_ollama(st.session_state.ollama_model, context, query)

        # 4. Generate TTS audio
        audio_path = generate_tts(text_response, st.session_state.tts_voice)

        return {
            "status": "success",
            "text_response": text_response,
            "audio_path": audio_path,
            "sources": list(sources)
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}

# ── Sidebar ───────────────────────────────────────────────────────

def sidebar_config():
    with st.sidebar:
        st.title("⚙️ Configuration")
        st.caption("100% local — no API keys needed")
        st.markdown("---")

        # Ollama model picker
        available_models = ["llama3.2", "mistral", "qwen2.5:7b"]
        try:
            fetched = ollama.list()
            if fetched and "models" in fetched:
                available_models = [m["model"] for m in fetched["models"]]
        except Exception:
            pass

        st.session_state.ollama_model = st.selectbox(
            "🤖 Ollama Model",
            options=available_models,
            help="Make sure Ollama is running: `ollama serve`"
        )

        # TTS voice picker
        voices = {
            "Jenny (US Female)": "en-US-JennyNeural",
            "Guy (US Male)": "en-US-GuyNeural",
            "Sonia (UK Female)": "en-GB-SoniaNeural",
            "Ryan (UK Male)": "en-GB-RyanNeural",
            "Natasha (AU Female)": "en-AU-NatashaNeural",
        }
        selected_voice_label = st.selectbox("🎤 TTS Voice", list(voices.keys()))
        st.session_state.tts_voice = voices[selected_voice_label]

        st.markdown("---")
        st.markdown("**Required local services:**")

        # Health checks
        qdrant_ok = False
        ollama_ok = False

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
            st.warning("Start Qdrant:\n```\ndocker run -d -p 6333:6333 qdrant/qdrant\n```")
        if not ollama_ok:
            st.warning("Start Ollama:\n```\nollama serve\n```")

# ── Main App ──────────────────────────────────────────────────────

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

    # ── File Upload Section ───────────────────────────────────────
    st.markdown("### 📂 Step 1: Upload Your Documentation")
    uploaded_files = st.file_uploader(
        "Drag & drop files here",
        type=["txt", "md", "pdf", "html", "htm", "docx"],
        accept_multiple_files=True,
        help="Supported: .txt, .md, .pdf, .html, .docx"
    )

    if uploaded_files:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(f"📎 {len(uploaded_files)} file(s) ready: {', '.join(f.name for f in uploaded_files)}")
        with col2:
            process_btn = st.button("⚡ Process Files", type="primary", use_container_width=True)

        if process_btn:
            with st.status("Processing files...", expanded=True) as status:
                try:
                    st.write("🔌 Connecting to Qdrant...")
                    client, emb_model = setup_qdrant()
                    st.session_state.client = client
                    st.session_state.embedding_model = emb_model
                    st.write("✅ Qdrant connected!")

                    pages = []
                    for f in uploaded_files:
                        st.write(f"📄 Parsing `{f.name}`...")
                        content = parse_uploaded_file(f)
                        if content.strip():
                            pages.append({"content": content, "source": f.name})
                        else:
                            st.warning(f"⚠️ Could not extract text from `{f.name}`")

                    st.write(f"💾 Embedding and storing {len(pages)} document(s)...")
                    total_chunks = store_embeddings(client, emb_model, pages)
                    st.write(f"✅ Stored {total_chunks} chunks across {len(pages)} files!")

                    st.session_state.setup_complete = True
                    st.session_state.docs_loaded = len(pages)
                    status.update(label="✅ Ready to answer questions!", state="complete")

                except Exception as e:
                    status.update(label="❌ Error during processing", state="error")
                    st.error(f"Error: {e}")

    st.markdown("---")

    # ── Query Section ─────────────────────────────────────────────
    st.markdown("### 💬 Step 2: Ask a Question")

    if not st.session_state.setup_complete:
        st.info("👆 Upload and process your files first!")
        return

    st.success(f"✅ {st.session_state.docs_loaded} document(s) loaded. Ask away!")

    query = st.text_input(
        "Your question:",
        placeholder="e.g., How do I authenticate API requests?",
    )

    if query:
        with st.status("Thinking...", expanded=True) as status:
            st.write(f"🔍 Searching docs with `{st.session_state.ollama_model}`...")
            result = process_query(query)

            if result["status"] == "success":
                status.update(label="✅ Done!", state="complete")

                st.markdown("### 💬 Answer")
                st.write(result["text_response"])

                st.markdown("### 🔊 Voice Response")
                st.audio(result["audio_path"], format="audio/mp3")

                with open(result["audio_path"], "rb") as f:
                    st.download_button(
                        "📥 Download Audio",
                        f.read(),
                        file_name="response.mp3",
                        mime="audio/mp3"
                    )

                st.markdown("### 📚 Sources Used")
                for s in result["sources"]:
                    st.markdown(f"- `{s}`")
            else:
                status.update(label="❌ Error", state="error")
                st.error(result.get("error", "Unknown error"))


if __name__ == "__main__":
    main()