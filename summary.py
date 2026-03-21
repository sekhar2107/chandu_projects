"""
╔══════════════════════════════════════════════════════════════════╗
║        DocSummarize  –  RAG Edition  (Single Script)             ║
║        FastAPI  +  Gemini AI  +  HTML / CSS / JS                 ║
╠══════════════════════════════════════════════════════════════════╣
║  INSTALL:                                                        ║
║    pip install fastapi uvicorn[standard] python-multipart        ║
║               google-generativeai PyPDF2 python-docx numpy       ║
║               python-dotenv                                      ║
║                                                                  ║
║  SETUP:                                                          ║
║    1. Create .env file with: GEMINI_API_KEY=your_key_here       ║
║    2. Run: python app.py                                         ║
║    3. Open: http://localhost:8001                                ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────────────────────────
import os
import io
import base64
import time
import uuid
import tempfile
import mimetypes
import logging
import hashlib
import re
from pathlib import Path
from typing import Any
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
import numpy as np
import google.generativeai as genai
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import PyPDF2
import docx
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load .env file
load_dotenv()

# ─────────────────────────────────────────────────────────────────
#  GEMINI SETUP
# ─────────────────────────────────────────────────────────────────
GEMINI_API_KEY = (os.getenv("GEMINI_API_KEY") or "").strip()
model = None
EMBED_MODEL = "models/gemini-embedding-001"
MODEL_CANDIDATES = [
    "gemini-2.5-flash",
    "models/gemini-2.5-flash",
    "gemini-2.0-flash",
    "models/gemini-2.0-flash",
    "gemini-1.5-flash",
    "models/gemini-1.5-flash",
    "gemini-1.5-pro",
    "models/gemini-1.5-pro",
]

if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY_HERE":
    try:
        genai.configure(api_key=GEMINI_API_KEY)

        # Try explicit preferred models first, then models discoverable from the account.
        discovered_models: list[str] = []
        try:
            for m in genai.list_models():
                methods = getattr(m, "supported_generation_methods", []) or []
                if "generateContent" in methods:
                    discovered_models.append(m.name)
        except Exception as e:
            logger.warning(f"⚠️ Could not list models from API: {e}")

        model_names = MODEL_CANDIDATES + discovered_models
        # Also try short names if API returns "models/<name>".
        model_names += [n.split("/", 1)[1] for n in discovered_models if n.startswith("models/")]

        # Keep order but remove duplicates.
        deduped_model_names = []
        seen = set()
        for name in model_names:
            if name and name not in seen:
                seen.add(name)
                deduped_model_names.append(name)

        model = None
        attempted = []
        for model_name in deduped_model_names:
            attempted.append(model_name)
            try:
                test_model = genai.GenerativeModel(model_name)
                test_response = test_model.generate_content("Reply only with: OK")
                if test_response:
                    model = test_model
                    logger.info(f"✅ Using model: {model_name}")
                    break
            except Exception as e:
                logger.warning(f"❌ Failed with {model_name}: {e}")

        if not model:
            raise Exception(f"Could not initialize any Gemini model. Tried: {attempted}")

        logger.info(f"✅ Gemini API Key loaded successfully")
    except Exception as e:
        logger.error(f"❌ Error configuring Gemini: {e}")
        model = None
else:
    logger.warning("\n" + "="*70)
    logger.warning("⚠️  WARNING: GEMINI_API_KEY not found!")
    logger.warning("\nMake sure your .env file contains:")
    logger.warning('  GEMINI_API_KEY=your_actual_key_here')
    logger.warning("="*70 + "\n")

# ─────────────────────────────────────────────────────────────────
#  MIME / EXTENSION GROUPS
# ─────────────────────────────────────────────────────────────────
IMAGE_MIMES = {"image/jpeg", "image/jpg", "image/png", "image/gif", "image/webp"}
AUDIO_MIMES = {"audio/mpeg", "audio/mp3", "audio/wav", "audio/ogg",
               "audio/flac", "audio/aac", "audio/x-m4a", "audio/mp4"}
VIDEO_MIMES = {"video/mp4", "video/quicktime", "video/x-msvideo",
               "video/x-matroska", "video/webm", "video/x-flv",
               "video/mpeg", "video/3gpp"}
AUDIO_EXTS = {".mp3", ".wav", ".ogg", ".flac", ".aac", ".m4a"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".mpeg", ".3gp"}

# ─────────────────────────────────────────────────────────────────
#  IN-MEMORY RAG VECTOR STORE
# ─────────────────────────────────────────────────────────────────
rag_store: dict[str, dict[str, Any]] = {}

CHUNK_SIZE = 400   # words per chunk
CHUNK_OVERLAP = 50  # words of overlap
EMBED_MAX_WORKERS = max(1, int(os.getenv("EMBED_MAX_WORKERS", "4")))
EMBED_CACHE_SIZE = max(0, int(os.getenv("EMBED_CACHE_SIZE", "512")))
GEMINI_MAX_RETRIES = max(1, int(os.getenv("GEMINI_MAX_RETRIES", "3")))
GEMINI_RETRY_BASE_SECONDS = float(os.getenv("GEMINI_RETRY_BASE_SECONDS", "1.5"))
RAG_RETRIEVE_TOP_K = max(1, int(os.getenv("RAG_RETRIEVE_TOP_K", "8")))
MAX_TRANSCRIPT_SUMMARY_CHARS = max(1000, int(os.getenv("MAX_TRANSCRIPT_SUMMARY_CHARS", "15000")))

# ─────────────────────────────────────────────────────────────────
#  RAG HELPERS
# ─────────────────────────────────────────────────────────────────
embedding_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()

def _cache_key(task: str, text: str) -> str:
    h = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
    return f"{task}:{h}"

def _cache_get(key: str) -> np.ndarray | None:
    if EMBED_CACHE_SIZE <= 0:
        return None
    vec = embedding_cache.get(key)
    if vec is not None:
        embedding_cache.move_to_end(key)
    return vec

def _cache_set(key: str, vec: np.ndarray) -> None:
    if EMBED_CACHE_SIZE <= 0:
        return
    embedding_cache[key] = vec
    embedding_cache.move_to_end(key)
    while len(embedding_cache) > EMBED_CACHE_SIZE:
        embedding_cache.popitem(last=False)

def is_transient_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    transient_signals = (
        "429", "rate limit", "resource_exhausted", "timeout",
        "timed out", "deadline", "temporar", "unavailable",
        "internal", "503", "502", "connection reset", "network",
    )
    return any(sig in msg for sig in transient_signals)

def with_retry(fn, *, action: str, retries: int = GEMINI_MAX_RETRIES):
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt >= retries or not is_transient_error(exc):
                break
            sleep_for = GEMINI_RETRY_BASE_SECONDS * (2 ** (attempt - 1))
            logger.warning(
                f"⚠️ {action} failed (attempt {attempt}/{retries}): {exc}. "
                f"Retrying in {sleep_for:.1f}s"
            )
            time.sleep(sleep_for)
    raise last_exc

def chunk_text(text: str) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        chunk = " ".join(words[start: start + CHUNK_SIZE])
        if chunk.strip():
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def get_embedding(text: str, task: str = "retrieval_document") -> np.ndarray:
    """Return a unit-normalised embedding vector from Gemini."""
    safe_text = (text or "")[:8000]
    key = _cache_key(task, safe_text)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    try:
        result = with_retry(
            lambda: genai.embed_content(
                model=EMBED_MODEL,
                content=safe_text,
                task_type=task,
            ),
            action=f"Embedding ({task})",
        )
        vec = np.array(result["embedding"], dtype=np.float32)
        norm = np.linalg.norm(vec)
        out = vec / norm if norm > 0 else vec
        _cache_set(key, out)
        return out
    except Exception as e:
        logger.error(f"❌ Error getting embedding: {e}")
        raise

def build_rag_index(doc_id: str, text: str) -> int:
    """Chunk the text, embed every chunk, store in rag_store."""
    logger.info(f"📄 Building RAG index for doc {doc_id[:8]}...")
    chunks = chunk_text(text)
    logger.info(f"   ✓ Created {len(chunks)} chunks")
    logger.info(f"   🔄 Embedding chunks...")
    if not chunks:
        rag_store[doc_id] = {"chunks": [], "embeddings": np.empty((0, 0), dtype=np.float32)}
        return 0

    if len(chunks) == 1 or EMBED_MAX_WORKERS == 1:
        embeddings = [get_embedding(c, task="retrieval_document") for c in chunks]
    else:
        with ThreadPoolExecutor(max_workers=min(EMBED_MAX_WORKERS, len(chunks))) as ex:
            embeddings = list(ex.map(lambda c: get_embedding(c, task="retrieval_document"), chunks))

    emb_matrix = np.vstack(embeddings).astype(np.float32)
    logger.info(f"   ✓ Embedded {len(embeddings)} chunks")
    rag_store[doc_id] = {"chunks": chunks, "embeddings": emb_matrix}
    return len(chunks)

def retrieve_top_k(doc_id: str, query: str, k: int = 5) -> list[str]:
    """Retrieve top-k most relevant chunks for a query."""
    if doc_id not in rag_store:
        return []
    q_vec = get_embedding(query, task="retrieval_query")
    store = rag_store[doc_id]
    emb_matrix = store["embeddings"]
    if emb_matrix.size == 0:
        return []
    scores = np.dot(emb_matrix, q_vec)
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    top_idx.sort()
    return [store["chunks"][i] for i in top_idx]

# ─────────────────────────────────────────────────────────────────
#  TEXT EXTRACTORS
# ─────────────────────────────────────────────────────────────────
def extract_pdf(data: bytes) -> str:
    reader = PyPDF2.PdfReader(io.BytesIO(data))
    return "\n\n".join(p.extract_text() or "" for p in reader.pages).strip()

def extract_docx(data: bytes) -> str:
    doc = docx.Document(io.BytesIO(data))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

# ─────────────────────────────────────────────────────────────────
#  MIME DETECTION
# ─────────────────────────────────────────────────────────────────
def detect_mime(file: UploadFile) -> str:
    mt = (file.content_type or "").split(";")[0].strip()
    if mt and mt != "application/octet-stream":
        return mt
    ext = Path(file.filename or "").suffix.lower()
    guessed, _ = mimetypes.guess_type(f"x{ext}")
    return guessed or "application/octet-stream"

def is_audio(mime: str, name: str) -> bool:
    return mime in AUDIO_MIMES or Path(name).suffix.lower() in AUDIO_EXTS

def is_video(mime: str, name: str) -> bool:
    return mime in VIDEO_MIMES or Path(name).suffix.lower() in VIDEO_EXTS

# ─────────────────────────────────────────────────────────────────
#  GEMINI WRAPPERS
# ─────────────────────────────────────────────────────────────────
def gemini_text(prompt: str) -> str:
    if not model:
        raise RuntimeError("Gemini model is not initialized.")
    try:
        response = with_retry(
            lambda: model.generate_content(prompt),
            action="Text generation",
        )
        return (response.text or "").strip()
    except Exception as e:
        logger.error(f"Gemini text generation error: {e}")
        raise

def gemini_inline(prompt: str, mime: str, data: bytes) -> str:
    if not model:
        raise RuntimeError("Gemini model is not initialized.")
    try:
        part = {"mime_type": mime, "data": base64.b64encode(data).decode()}
        response = with_retry(
            lambda: model.generate_content([prompt, part]),
            action="Inline multimodal generation",
        )
        return (response.text or "").strip()
    except Exception as e:
        logger.error(f"Gemini inline generation error: {e}")
        raise

def gemini_upload(prompt: str, mime: str, data: bytes, suffix: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    
    uploaded = None
    try:
        uploaded = with_retry(
            lambda: genai.upload_file(tmp_path, mime_type=mime),
            action="File upload",
        )
        # Wait for processing
        for _ in range(60):  # Increased timeout
            file_state = with_retry(
                lambda: genai.get_file(uploaded.name),
                action="File state check",
            )
            if file_state.state.name == "ACTIVE":
                break
            if file_state.state.name == "FAILED":
                raise RuntimeError("Gemini rejected the file.")
            time.sleep(2)
        else:
            raise RuntimeError("Gemini file processing timed out.")
        
        response = with_retry(
            lambda: model.generate_content([prompt, uploaded]),
            action="Uploaded file generation",
        )
        result = (response.text or "").strip()
        
        # Cleanup
        try:
            if uploaded:
                genai.delete_file(uploaded.name)
        except Exception:
            pass
        
        return result
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

# ─────────────────────────────────────────────────────────────────
#  PROMPTS
# ─────────────────────────────────────────────────────────────────
RAG_SUMMARY_PROMPT = """You are a professional document analyst.
Using ONLY the document excerpts below, write a comprehensive, well-structured summary.

Include:
  • Main topics and themes
  • Key points and important details
  • Conclusions or action items (if any)

Be clear, concise, and professional. Do not fabricate information not present in the excerpts.
Do not include any symbols like *,#,& etc
Document Excerpts:
{context}

Summary:"""

TRANSCRIBE_AUDIO = "Transcribe this audio accurately and completely. Return ONLY the transcript. No timestamps, no labels, no commentary."
TRANSCRIBE_VIDEO = "Transcribe all spoken dialogue and narration from this video. Return ONLY the transcript. No timestamps, no labels, no commentary."

IMAGE_SUMMARY = """Analyze this image and provide a detailed summary:
  • What is shown (objects, people, scenes)
  • Any visible text or data
  • Context and likely purpose
  • Key takeaways
Be clear and well-structured."""

SUMMARIZE_TRANSCRIPT = """Provide a comprehensive summary of this transcript.
Include main topics, key points, and conclusions.
Do not include any symbols like *,#,& etc
Transcript:
{content}"""

decode_text = lambda b: b.decode("utf-8", errors="replace")
TEXT_DOC_CFG = ("text file", decode_text, "text_rag")
WORD_DOC_CFG = ("Word document", extract_docx, "docx_rag")
DOC_MIME_MAP = {
    "application/pdf": ("PDF", extract_pdf, "pdf_rag"),
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": WORD_DOC_CFG,
    "application/msword": WORD_DOC_CFG,
    "text/plain": TEXT_DOC_CFG,
    "text/markdown": TEXT_DOC_CFG,
}
DOC_EXT_MAP = {".doc": WORD_DOC_CFG, ".docx": WORD_DOC_CFG, ".md": TEXT_DOC_CFG, ".txt": TEXT_DOC_CFG}

def summarize_transcript_text(transcript: str) -> str:
    return clean_summary_text(
        gemini_text(SUMMARIZE_TRANSCRIPT.format(content=transcript[:MAX_TRANSCRIPT_SUMMARY_CHARS]))
    )

def process_av_file(mode: str, transcript_prompt: str, fallback_mime: str, mime: str, filename: str, data: bytes, ext: str) -> dict:
    logger.info(f"Detected as {mode} file")
    allowed = VIDEO_MIMES if mode == "video" else AUDIO_MIMES
    real_mime = mime if mime in allowed else (mimetypes.guess_type(filename)[0] or fallback_mime)
    transcript = gemini_upload(transcript_prompt, real_mime, data, ext or (".mp4" if mode == "video" else ".mp3"))
    if not transcript:
        raise HTTPException(422, f"Could not extract transcript from this {mode} file.")
    summary = summarize_transcript_text(transcript)
    logger.info(f"✓ {mode.capitalize()} processed successfully")
    return {"mode": mode, "transcript": transcript, "summary": summary}

def process_text_like_document(label: str, extractor, mode: str, data: bytes) -> dict:
    text = extractor(data)
    if not text:
        raise HTTPException(422, f"Could not extract text from this {label}.")
    logger.info(f"   ✓ Extracted {len(text)} characters")
    result = process_document_rag(text)
    result["mode"] = mode
    logger.info(f"✓ {label.capitalize()} processed successfully")
    return result

def clean_summary_text(text: str) -> str:
    """Remove markdown/special symbols to keep output plain and clean."""
    if not text:
        return ""
    cleaned = text
    # Remove common markdown decoration symbols.
    cleaned = cleaned.replace("**", "").replace("*", "").replace("#", "")
    cleaned = cleaned.replace("`", "").replace("_", "")
    # Replace bullet symbols with plain list marker.
    cleaned = cleaned.replace("•", "-").replace("◦", "-").replace("▪", "-")
    # Remove repeated punctuation decoration.
    cleaned = re.sub(r"[~^|]+", "", cleaned)
    # Collapse excessive blank lines/spaces.
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    # Clean markdown-style heading/list prefixes left at line start.
    cleaned = re.sub(r"^\s*[-]{2,}\s*", "- ", cleaned, flags=re.MULTILINE)
    return cleaned.strip()

# ─────────────────────────────────────────────────────────────────
#  DOCUMENT RAG PROCESSOR
# ─────────────────────────────────────────────────────────────────
def process_document_rag(text: str) -> dict:
    """Full RAG pipeline for a document."""
    if not text or not text.strip():
        raise HTTPException(422, "Document text is empty after extraction.")

    doc_id = str(uuid.uuid4())
    chunk_count = build_rag_index(doc_id, text)
    if chunk_count == 0:
        raise HTTPException(422, "Document does not contain enough text to summarize.")

    summary_query = "What are the main topics, key arguments, important facts, conclusions, and recommendations in this document?"
    top_chunks = retrieve_top_k(doc_id, summary_query, k=min(RAG_RETRIEVE_TOP_K, chunk_count))
    context = "\n\n---\n\n".join(top_chunks)
    summary = clean_summary_text(gemini_text(RAG_SUMMARY_PROMPT.format(context=context)))

    return {
        "doc_id": doc_id,
        "summary": summary,
        "chunk_count": chunk_count,
    }

# ─────────────────────────────────────────────────────────────────
#  CORE FILE PROCESSOR
# ─────────────────────────────────────────────────────────────────
async def process_file(file: UploadFile) -> dict:
    data = await file.read()
    mime = detect_mime(file)
    filename = file.filename or "file"
    ext = Path(filename).suffix.lower()
    
    logger.info(f"📥 Processing file: {filename} ({len(data)} bytes, type: {mime})")

    if is_audio(mime, filename):
        return process_av_file("audio", TRANSCRIBE_AUDIO, "audio/mpeg", mime, filename, data, ext)
    if is_video(mime, filename):
        return process_av_file("video", TRANSCRIBE_VIDEO, "video/mp4", mime, filename, data, ext)

    # IMAGE
    if mime in IMAGE_MIMES:
        logger.info(f"🖼️ Detected as image file")
        safe_mime = "image/jpeg" if mime == "image/jpg" else mime
        summary = clean_summary_text(gemini_inline(IMAGE_SUMMARY, safe_mime, data))
        logger.info(f"✓ Image processed successfully")
        return {"mode": "image", "summary": summary}

    # PDF (with scanned fallback)
    if mime == "application/pdf":
        logger.info(f"📄 Detected as PDF")
        text = extract_pdf(data)
        if text and len(text) > 50:
            return process_text_like_document("pdf", lambda _: text, "pdf_rag", data)
        logger.info(f"⚠️ PDF appears to be scanned, using vision")
        summary = clean_summary_text(gemini_inline("Summarize this PDF document comprehensively.", "application/pdf", data))
        logger.info(f"✓ PDF processed successfully")
        return {"mode": "document", "summary": summary}

    # DOC / DOCX / TXT / MD
    doc_type = DOC_MIME_MAP.get(mime) or DOC_EXT_MAP.get(ext)
    if doc_type:
        label, extractor, mode = doc_type
        logger.info(f"📝 Detected as {label}")
        return process_text_like_document(label, extractor, mode, data)

    raise HTTPException(415, f"Unsupported file type: '{mime}' (extension: '{ext}')")

# ─────────────────────────────────────────────────────────────────
#  FASTAPI APP
# ─────────────────────────────────────────────────────────────────
app = FastAPI(title="DocSummarize RAG", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(content=HTML)

@app.post("/summarize")
async def summarize(file: UploadFile = File(...)):
    if not GEMINI_API_KEY:
        raise HTTPException(500, "GEMINI_API_KEY not configured. Set the environment variable and restart.")
    if not model:
        raise HTTPException(500, "Gemini model not initialized. Check your API key.")
    
    try:
        logger.info(f"Processing file: {file.filename}")
        result = await process_file(file)
        logger.info(f"Successfully processed {file.filename}")
        return JSONResponse(result)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error processing file {file.filename}: {str(exc)}", exc_info=True)
        raise HTTPException(500, f"Processing error: {str(exc)}")

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": model.model_name if model else "None",
        "embed_model": EMBED_MODEL,
        "docs_indexed": len(rag_store),
        "embed_workers": EMBED_MAX_WORKERS,
        "embed_cache_size": EMBED_CACHE_SIZE,
        "embed_cache_entries": len(embedding_cache),
        "gemini_retries": GEMINI_MAX_RETRIES,
    }

# ─────────────────────────────────────────────────────────────────
#  HTML / CSS / JS FRONTEND
# ─────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>DocSummarize · RAG Edition</title>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet"/>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%}
body{font-family:'Plus Jakarta Sans',sans-serif;background:#F1F5F9;color:#1A202C;display:flex;min-height:100vh}

/* Sidebar */
.sidebar{width:240px;min-height:100vh;background:#0F172A;display:flex;flex-direction:column;position:fixed;left:0;top:0;bottom:0;z-index:10}
.sb-logo{padding:20px 18px 16px;border-bottom:1px solid rgba(255,255,255,.06);display:flex;align-items:center;gap:10px}
.sb-logo-icon{width:36px;height:36px;background:linear-gradient(135deg,#6366F1,#8B5CF6);border-radius:9px;display:flex;align-items:center;justify-content:center;flex-shrink:0}
.sb-name{font-weight:800;font-size:15px;color:#F8FAFC}
.sb-sub{font-size:11px;color:#4B5563;margin-top:1px}
.nav-lbl{padding:14px 18px 6px;font-size:10px;font-weight:700;color:#374151;letter-spacing:.12em;text-transform:uppercase}
.nav-link{display:flex;align-items:center;gap:10px;padding:9px 16px;margin:2px 10px;font-size:13px;font-weight:500;color:#94A3B8;border-radius:8px;cursor:pointer;transition:all .15s;text-decoration:none}
.nav-link.active{background:rgba(99,102,241,.2);color:#A5B4FC}
.nav-link:hover:not(.active){background:rgba(255,255,255,.05);color:#CBD5E0}
.nav-link svg{opacity:.7;flex-shrink:0}
.fmt-panel{margin:12px;background:rgba(99,102,241,.1);border:1px solid rgba(99,102,241,.2);border-radius:10px;padding:14px 16px}
.fmt-panel-title{font-size:11px;font-weight:700;color:#818CF8;margin-bottom:8px;text-transform:uppercase;letter-spacing:.08em}
.fmt-tags{display:flex;flex-wrap:wrap;gap:5px}
.fmt-tag{background:rgba(99,102,241,.2);color:#A5B4FC;font-size:10px;font-weight:700;padding:2px 7px;border-radius:4px}
.fmt-tag.av{background:rgba(245,158,11,.2);color:#F59E0B}

/* Main */
.main{margin-left:240px;flex:1;display:flex;flex-direction:column;min-height:100vh}
.topbar{background:white;border-bottom:1px solid #E2E8F0;padding:0 28px;height:60px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:5;box-shadow:0 1px 3px rgba(0,0,0,.04)}
.tb-h1{font-weight:700;font-size:16px;color:#0F172A}
.tb-sub{font-size:12px;color:#94A3B8;margin-top:1px}
.tb-acts{display:flex;gap:10px;align-items:center}

/* Buttons */
.btn{display:inline-flex;align-items:center;gap:6px;border:none;border-radius:8px;font-size:13px;font-weight:600;cursor:pointer;transition:all .15s;font-family:inherit;white-space:nowrap;padding:8px 18px}
.btn-primary{background:#6366F1;color:white}
.btn-primary:hover{background:#4F46E5;box-shadow:0 4px 12px rgba(99,102,241,.3)}
.btn-primary:disabled{background:#A5B4FC;cursor:not-allowed;box-shadow:none}
.btn-outline{background:white;color:#6366F1;border:1.5px solid #C7D2FE}
.btn-outline:hover{background:#EEF2FF;border-color:#6366F1}
.btn-sm{background:#EEF2FF;color:#4F46E5;border:none;padding:5px 12px;border-radius:6px;font-size:12px;font-weight:600;cursor:pointer;transition:all .15s;font-family:inherit}
.btn-sm:hover{background:#C7D2FE}
.btn-del{background:none;border:none;cursor:pointer;padding:6px;border-radius:6px;color:#94A3B8;display:flex;align-items:center;transition:all .15s}
.btn-del:hover{background:#FEE2E2;color:#DC2626}

/* Content */
.content{padding:24px 28px;flex:1}
.stats-row{display:flex;gap:14px;margin-bottom:22px}
.stat-card{background:white;border-radius:12px;border:1px solid #E2E8F0;padding:18px 20px;flex:1;box-shadow:0 1px 3px rgba(0,0,0,.04);border-top:3px solid transparent}
.stat-num{font-size:26px;font-weight:800}
.stat-lbl{font-size:12px;color:#64748B;font-weight:500;margin-top:2px}

/* Drop zone */
.drop-zone{border:2px dashed #CBD5E0;border-radius:14px;background:white;padding:44px 24px;text-align:center;cursor:pointer;transition:all .2s;margin-bottom:22px}
.drop-zone:hover,.drop-zone.over{border-color:#6366F1;background:#EEF2FF}
.dz-icon{width:54px;height:54px;background:#EEF2FF;border-radius:14px;display:flex;align-items:center;justify-content:center;margin:0 auto 14px}
.dz-title{font-weight:700;font-size:16px;color:#0F172A;margin-bottom:5px}
.dz-sub{font-size:13px;color:#64748B;margin-bottom:14px}
.chips{display:flex;gap:6px;justify-content:center;flex-wrap:wrap}
.chip{display:inline-flex;align-items:center;background:#EEF2FF;color:#4338CA;font-size:11px;font-weight:700;padding:3px 10px;border-radius:6px}
.chip.av{background:#FFF3CD;color:#856404}
.chip.rag{background:#ECFDF5;color:#065F46}

/* File list */
.fl-hdr{display:flex;justify-content:space-between;align-items:center;margin-bottom:14px}
.fl-title{font-weight:700;font-size:15px;color:#0F172A}
.fl-count{color:#94A3B8;font-weight:500}
.btn-clear{background:none;border:none;color:#94A3B8;font-size:13px;cursor:pointer;font-family:inherit;font-weight:500}
.btn-clear:hover{color:#DC2626}

/* File card */
.file-card{background:white;border-radius:12px;border:1px solid #E2E8F0;box-shadow:0 1px 3px rgba(0,0,0,.04);overflow:hidden;margin-bottom:10px;transition:box-shadow .2s,border-color .2s;animation:slideIn .25s ease}
.file-card:hover{box-shadow:0 4px 16px rgba(99,102,241,.1);border-color:#C7D2FE}
@keyframes slideIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
.file-hdr{display:flex;align-items:center;gap:14px;padding:16px 20px}
.type-badge{width:42px;height:42px;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:800;color:white;letter-spacing:.05em;flex-shrink:0}
.file-info{flex:1;min-width:0}
.file-name{font-weight:600;font-size:14px;color:#0F172A;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.file-meta{font-size:12px;color:#94A3B8;margin-top:3px}
.file-acts{display:flex;gap:8px;align-items:center}

/* Pills */
.pill{display:inline-flex;align-items:center;gap:5px;padding:4px 10px;border-radius:20px;font-size:11px;font-weight:600;white-space:nowrap}
.p-idle{background:#F1F5F9;color:#475569}
.p-load{background:#FEF3C7;color:#92400E;animation:blink 1.4s infinite}
.p-done{background:#DCFCE7;color:#166534}
.p-err{background:#FEE2E2;color:#991B1B}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.5}}

/* Error styling */
.error-details{font-size:12px;color:#991B1B;margin-top:8px;padding:8px;background:#FEE2E2;border-radius:6px;border-left:3px solid #DC2626}
.retry-badge{display:inline-flex;align-items:center;gap:4px;background:#FEF3C7;color:#92400E;font-size:11px;padding:2px 8px;border-radius:12px;margin-left:8px}
.timeout-warning{color:#B45309;font-size:11px;margin-top:4px;display:flex;align-items:center;gap:4px}

/* Card body */
.card-body{border-top:1px solid #F1F5F9}
.tabs{display:flex;border-bottom:1px solid #E2E8F0;padding:0 20px;background:#FAFBFF}
.tab{padding:10px 16px;font-size:13px;font-weight:600;color:#94A3B8;cursor:pointer;border-bottom:2px solid transparent;transition:all .15s;margin-bottom:-1px}
.tab.active{color:#6366F1;border-bottom-color:#6366F1}
.tab:hover:not(.active){color:#4F46E5}
.tab-panel{display:none;padding:20px;background:#FAFBFF;animation:fd .2s ease}
.tab-panel.active{display:block}
@keyframes fd{from{opacity:0;transform:translateY(-4px)}to{opacity:1;transform:translateY(0)}}
.p-hdr{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px}
.p-lbl{font-weight:700;font-size:13px;color:#374151;display:flex;align-items:center;gap:7px}
.p-icon{width:20px;height:20px;background:#EEF2FF;border-radius:5px;display:flex;align-items:center;justify-content:center}
.content-box{background:white;border:1px solid #E2E8F0;border-radius:10px;padding:18px 20px;font-size:14px;line-height:1.8;color:#374151;white-space:pre-wrap;max-height:420px;overflow-y:auto}
.content-box::-webkit-scrollbar{width:5px}
.content-box::-webkit-scrollbar-thumb{background:#CBD5E0;border-radius:3px}
.err-area{padding:14px 20px;background:#FFF5F5;font-size:13px;color:#B91C1C;display:flex;gap:8px;align-items:flex-start}

/* RAG badge */
.rag-badge{display:inline-flex;align-items:center;gap:6px;background:#ECFDF5;color:#065F46;border:1px solid #A7F3D0;font-size:11px;font-weight:700;padding:4px 10px;border-radius:6px;margin-bottom:12px}

/* Progress */
.prog-wrap{padding:16px 20px;background:#FAFBFF}
.prog-lbl{font-size:12px;font-weight:600;color:#374151;margin-bottom:8px}
.prog-bar{height:6px;background:#E2E8F0;border-radius:6px;overflow:hidden}
.prog-fill{height:100%;border-radius:6px;background:linear-gradient(90deg,#6366F1,#8B5CF6,#6366F1);background-size:200% 100%;animation:sweep 2s linear infinite}
@keyframes sweep{0%{background-position:200% 0}100%{background-position:-200% 0}}
.steps{display:flex;align-items:center;gap:8px;padding:12px 20px;background:#F8FAFF;border-top:1px solid #F1F5F9}
.step{display:flex;align-items:center;gap:6px;font-size:12px;font-weight:600;color:#94A3B8}
.step.done{color:#059669}.step.active{color:#6366F1}
.step-dot{width:8px;height:8px;border-radius:50%;background:#E2E8F0;flex-shrink:0}
.step.done .step-dot{background:#059669}
.step.active .step-dot{background:#6366F1;animation:pd 1s ease-in-out infinite}
@keyframes pd{0%,100%{transform:scale(1)}50%{transform:scale(1.5);opacity:.6}}
.step-arr{color:#CBD5E0;font-size:13px}

/* Empty state */
.empty{text-align:center;padding:40px 0}
.empty .ico{font-size:42px;margin-bottom:12px}
.empty .ttl{font-weight:600;font-size:15px;color:#64748B}
.empty .sub{font-size:13px;color:#94A3B8;margin-top:4px}

/* Footer */
.footer{border-top:1px solid #E2E8F0;background:white;padding:12px 28px;display:flex;justify-content:space-between}
.footer span{font-size:12px;color:#94A3B8}

#file-input{display:none}
@media(max-width:768px){.sidebar{display:none}.main{margin-left:0}.stats-row{flex-wrap:wrap}.stat-card{min-width:calc(50% - 7px)}}
</style>
</head>
<body>

<!-- Sidebar -->
<aside class="sidebar">
  <div class="sb-logo">
    <div class="sb-logo-icon">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
        <polyline points="14 2 14 8 20 8"/>
        <line x1="16" y1="13" x2="8" y2="13"/>
        <line x1="16" y1="17" x2="8" y2="17"/>
      </svg>
    </div>
    <div><div class="sb-name">DocSummarize</div><div class="sb-sub">RAG Edition · Gemini AI</div></div>
  </div>
  <nav style="padding:14px 0;flex:1">
    <div class="nav-lbl">Navigation</div>
    <a class="nav-link active" href="#" onclick="switchView('dashboard'); return false;">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>
      Dashboard
    </a>
    <a class="nav-link" href="#" onclick="switchView('dashboard'); return false;">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="16 16 12 12 8 16"/><line x1="12" y1="12" x2="12" y2="21"/><path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3"/></svg>
      Upload
    </a>
    <a class="nav-link" href="#" onclick="switchView('history'); return false;">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/></svg>
      History
    </a>
  </nav>
  <div class="fmt-panel">
    <div class="fmt-panel-title">Supported Formats</div>
    <div class="fmt-tags">
      <span class="fmt-tag" title="RAG-powered">PDF ✦</span>
      <span class="fmt-tag" title="RAG-powered">DOCX ✦</span>
      <span class="fmt-tag" title="RAG-powered">TXT ✦</span>
      <span class="fmt-tag" title="RAG-powered">MD ✦</span>
      <span class="fmt-tag">JPG</span><span class="fmt-tag">PNG</span>
      <span class="fmt-tag">GIF</span><span class="fmt-tag">WebP</span>
      <span class="fmt-tag av">MP3</span><span class="fmt-tag av">WAV</span>
      <span class="fmt-tag av">OGG</span><span class="fmt-tag av">M4A</span>
      <span class="fmt-tag av">MP4</span><span class="fmt-tag av">MOV</span>
      <span class="fmt-tag av">AVI</span><span class="fmt-tag av">MKV</span>
    </div>
    <div style="font-size:10px;color:#4B5563;margin-top:8px">✦ RAG-powered summarization</div>
  </div>
</aside>

<!-- Main -->
<div class="main">
  <header class="topbar">
    <div>
      <div class="tb-h1">Document Summarizer</div>
      <div class="tb-sub">RAG-powered summaries · Documents chunked & embedded with Gemini</div>
    </div>
    <div class="tb-acts">
      <button class="btn btn-primary" id="btn-all" style="display:none">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"/></svg>
        <span id="btn-all-lbl">Summarize All</span>
      </button>
      <button class="btn btn-outline" onclick="document.getElementById('file-input').click()">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
        Upload Files
      </button>
      <input type="file" id="file-input" multiple
        accept=".pdf,.jpg,.jpeg,.png,.gif,.webp,.docx,.doc,.txt,.md,.mp3,.wav,.ogg,.flac,.aac,.m4a,.mp4,.mov,.avi,.mkv,.webm,.flv"
        onchange="handleInput(event)"/>
    </div>
  </header>

  <div class="content">
    <div id="dashboard-view">
    <!-- Stats -->
    <div class="stats-row" id="stats-row" style="display:none">
      <div class="stat-card" style="border-top-color:#6366F1"><div class="stat-num" id="s-total" style="color:#6366F1">0</div><div class="stat-lbl">Total Files</div></div>
      <div class="stat-card" style="border-top-color:#059669"><div class="stat-num" id="s-done"  style="color:#059669">0</div><div class="stat-lbl">Completed</div></div>
      <div class="stat-card" style="border-top-color:#D97706"><div class="stat-num" id="s-proc"  style="color:#D97706">0</div><div class="stat-lbl">Processing</div></div>
      <div class="stat-card" style="border-top-color:#64748B"><div class="stat-num" id="s-idle"  style="color:#64748B">0</div><div class="stat-lbl">Pending</div></div>
    </div>

    <!-- Drop zone -->
    <div class="drop-zone" id="dz"
      onclick="document.getElementById('file-input').click()"
      ondragover="dzOver(event)" ondragleave="dzLeave()" ondrop="dzDrop(event)">
      <div class="dz-icon">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#6366F1" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
          <polyline points="16 16 12 12 8 16"/><line x1="12" y1="12" x2="12" y2="21"/>
          <path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3"/>
        </svg>
      </div>
      <div class="dz-title">Drag & drop files here</div>
      <div class="dz-sub">Documents get full RAG indexing — summarize & ask questions</div>
      <div class="chips">
        <span class="chip rag">🔍 PDF · DOCX · TXT (RAG-powered)</span>
        <span class="chip">JPG · PNG</span>
        <span class="chip av">🎵 MP3 · WAV</span>
        <span class="chip av">🎬 MP4 · MOV</span>
      </div>
    </div>

    <!-- File list -->
    <div id="fl-sec" style="display:none">
      <div class="fl-hdr">
        <div class="fl-title">Uploaded Files <span class="fl-count" id="fl-count"></span></div>
        <button class="btn-clear" onclick="clearAll()">Clear All</button>
      </div>
      <div id="cards"></div>
    </div>

    <div class="empty" id="empty">
      <div class="ico">📂</div>
      <div class="ttl">No files uploaded yet</div>
      <div class="sub">Upload a document, image, audio, or video to get started</div>
    </div>
    </div>
    <div id="history-view" style="display:none">
      <div class="fl-hdr" style="margin-bottom:16px">
        <div class="fl-title">History <span class="fl-count" id="history-count"></span></div>
      </div>
      <div id="history-list"></div>
      <div class="empty" id="history-empty" style="display:none">
        <div class="ico">H</div>
        <div class="ttl">No history yet</div>
        <div class="sub">Run at least one summary to see it here</div>
      </div>
    </div>
  </div>

  <footer class="footer">
    <span>DocSummarize · RAG Edition · Gemini 1.5 Flash + text-embedding-004</span>
    <span>✦ PDF · DOCX · TXT · MD get full RAG vector search & summarization</span>
  </footer>
</div>

<script>
/* State */
const files = [];
let expandedId = null;
let currentView = "dashboard";
const HISTORY_KEY = "docsummarize_history_v1";
let historyItems = loadHistory();

/* Type map */
const TM = {
  "application/pdf":{label:"PDF Document",badge:"PDF",color:"#E53E3E",kind:"doc"},
  "image/jpeg":{label:"JPEG Image",badge:"IMG",color:"#38A169",kind:"img"},
  "image/jpg":{label:"JPG Image",badge:"IMG",color:"#38A169",kind:"img"},
  "image/png":{label:"PNG Image",badge:"IMG",color:"#38A169",kind:"img"},
  "image/gif":{label:"GIF Image",badge:"IMG",color:"#38A169",kind:"img"},
  "image/webp":{label:"WebP Image",badge:"IMG",color:"#38A169",kind:"img"},
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
    {label:"Word Document",badge:"DOC",color:"#2B6CB0",kind:"doc"},
  "application/msword":{label:"Word Document",badge:"DOC",color:"#2B6CB0",kind:"doc"},
  "text/plain":{label:"Text File",badge:"TXT",color:"#6B46C1",kind:"doc"},
  "text/markdown":{label:"Markdown",badge:"MD",color:"#6B46C1",kind:"doc"},
  "audio/mpeg":{label:"MP3 Audio",badge:"MP3",color:"#D97706",kind:"audio"},
  "audio/wav":{label:"WAV Audio",badge:"WAV",color:"#D97706",kind:"audio"},
  "audio/ogg":{label:"OGG Audio",badge:"OGG",color:"#D97706",kind:"audio"},
  "audio/flac":{label:"FLAC Audio",badge:"FLAC",color:"#D97706",kind:"audio"},
  "audio/aac":{label:"AAC Audio",badge:"AAC",color:"#D97706",kind:"audio"},
  "audio/x-m4a":{label:"M4A Audio",badge:"M4A",color:"#D97706",kind:"audio"},
  "video/mp4":{label:"MP4 Video",badge:"MP4",color:"#DD6B20",kind:"video"},
  "video/quicktime":{label:"MOV Video",badge:"MOV",color:"#DD6B20",kind:"video"},
  "video/x-msvideo":{label:"AVI Video",badge:"AVI",color:"#DD6B20",kind:"video"},
  "video/x-matroska":{label:"MKV Video",badge:"MKV",color:"#DD6B20",kind:"video"},
  "video/webm":{label:"WebM Video",badge:"WEBM",color:"#DD6B20",kind:"video"},
};
const AUDIO_EXTS=new Set([".mp3",".wav",".ogg",".flac",".aac",".m4a"]);
const VIDEO_EXTS=new Set([".mp4",".mov",".avi",".mkv",".webm",".flv",".mpeg",".3gp"]);

function typeInfo(f){
  const ext=(f.name.match(/\.[^.]+$/)||[""])[0].toLowerCase();
  if(ext===".docx"||ext===".doc") return TM["application/vnd.openxmlformats-officedocument.wordprocessingml.document"];
  const t=TM[f.type]; if(t) return t;
  if(AUDIO_EXTS.has(ext)) return {label:"Audio File",badge:"AUD",color:"#D97706",kind:"audio"};
  if(VIDEO_EXTS.has(ext)) return {label:"Video File",badge:"VID",color:"#DD6B20",kind:"video"};
  return {label:"File",badge:"???",color:"#718096",kind:"doc"};
}
const fmtSz=b=>b<1024?b+" B":b<1048576?(b/1024).toFixed(1)+" KB":(b/1048576).toFixed(1)+" MB";
const esc=s=>String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");

function loadHistory(){
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}
function saveHistory(){
  try { localStorage.setItem(HISTORY_KEY, JSON.stringify(historyItems)); } catch {}
}
function addHistoryItem(f){
  historyItems.unshift({
    id: Date.now().toString(36) + Math.random().toString(36).slice(2, 8),
    ts: new Date().toISOString(),
    name: f.name,
    size: f.size,
    type: f.type || "",
    summary: f.summary || "",
    transcript: f.transcript || "",
    chunk_count: f.chunk_count || 0,
    doc_id: f.doc_id || null
  });
  if(historyItems.length > 100) historyItems = historyItems.slice(0, 100);
  saveHistory();
}
function switchView(view){
  currentView = view === "history" ? "history" : "dashboard";
  const nav = document.querySelectorAll(".nav-link");
  nav.forEach((el, idx) => {
    const active = currentView === "history" ? idx === 2 : idx === 0;
    el.classList.toggle("active", active);
  });
  render();
}
function renderHistory(){
  const list = document.getElementById("history-list");
  const empty = document.getElementById("history-empty");
  const count = document.getElementById("history-count");
  count.textContent = `(${historyItems.length})`;
  if(historyItems.length === 0){
    list.innerHTML = "";
    empty.style.display = "block";
    return;
  }
  empty.style.display = "none";
  list.innerHTML = historyItems.map(h => {
    const dt = new Date(h.ts);
    const ts = isNaN(dt.getTime()) ? "Unknown time" : dt.toLocaleString();
    const rag = h.doc_id ? ` · RAG ${h.chunk_count} chunks` : "";
    return `<div class="file-card">
      <div class="file-hdr">
        <div class="type-badge" style="background:#6366F1">HIS</div>
        <div class="file-info">
          <div class="file-name" title="${esc(h.name)}">${esc(h.name)}</div>
          <div class="file-meta">${fmtSz(h.size)} · ${esc(ts)}${rag}</div>
        </div>
      </div>
      <div class="card-body">
        <div class="tab-panel active" style="display:block">
          <div class="p-hdr">
            <div class="p-lbl"><div class="p-icon">S</div>AI Summary</div>
          </div>
          <div class="content-box">${esc(h.summary || "No summary stored.")}</div>
        </div>
      </div>
    </div>`;
  }).join("");
}

/* Drag & drop */
function dzOver(e){e.preventDefault();document.getElementById("dz").classList.add("over");}
function dzLeave(){document.getElementById("dz").classList.remove("over");}
function dzDrop(e){e.preventDefault();document.getElementById("dz").classList.remove("over");addFiles(e.dataTransfer.files);}
function handleInput(e){addFiles(e.target.files);e.target.value="";}
function addFiles(list){
  Array.from(list).forEach(f=>files.push({
    id:Math.random().toString(36).slice(2),file:f,name:f.name,size:f.size,type:f.type,
    status:"idle",summary:null,transcript:null,error:null,
    doc_id:null,chunk_count:0,
    activeTab:"summary"
  }));
  render();
}

/* File actions */
function removeFile(id){const i=files.findIndex(f=>f.id===id);if(i>-1)files.splice(i,1);if(expandedId===id)expandedId=null;render();}
function clearAll(){files.length=0;expandedId=null;render();}
function toggleExpand(id){expandedId=(expandedId===id)?null:id;render();}
function setTab(id,tab){const f=files.find(x=>x.id===id);if(f){f.activeTab=tab;render();}}

/* Summarize with timeout and retry */
async function summarize(id){
  const f = files.find(x => x.id === id);
  if(!f) return;
  
  f.status = "loading"; 
  f.error = null;
  render();
  
  // Create abort controller for timeout
  const controller = new AbortController();
  const timeoutId = setTimeout(() => {
    controller.abort();
    console.log(`Request timeout for file: ${f.name}`);
  }, 120000); // 2 minute timeout
  
  try {
    const fd = new FormData(); 
    fd.append("file", f.file);
    
    // Retry logic
    let retries = 3;
    let lastError = null;
    
    while (retries > 0) {
      try {
        const res = await fetch("/summarize", {
          method: "POST",
          body: fd,
          signal: controller.signal,
          headers: {
            'Accept': 'application/json'
          }
        });
        
        clearTimeout(timeoutId);
        
        if (!res.ok) {
          const errorData = await res.json().catch(() => ({}));
          throw new Error(errorData.detail || `Server error: ${res.status}`);
        }
        
        const data = await res.json();
        
        if (!data || typeof data !== 'object') {
          throw new Error('Invalid response from server');
        }
        
        f.status = "done";
        f.summary = data.summary || "No summary generated";
        f.transcript = data.transcript || null;
        f.doc_id = data.doc_id || null;
        f.chunk_count = data.chunk_count || 0;
        f.activeTab = "summary";
        addHistoryItem(f);
        expandedId = id;
        
        // Success - break retry loop
        break;
        
      } catch (err) {
        lastError = err;
        retries--;
        
        if (retries > 0) {
          // Wait before retrying (exponential backoff)
          await new Promise(resolve => setTimeout(resolve, 2000 * (3 - retries)));
          console.log(`Retrying... ${retries} attempts left for ${f.name}`);
          
          // Update UI to show retry status
          f.status = "loading";
          f.error = `Retrying... (${3-retries}/3)`;
          render();
        }
      }
    }
    
    // If all retries failed
    if (retries === 0 && lastError) {
      throw lastError;
    }
    
  } catch(err) {
    clearTimeout(timeoutId);
    
    f.status = "error";
    
    // Handle different error types
    if (err.name === 'AbortError') {
      f.error = 'Request timeout - file may be too large or server busy';
    } else if (err.name === 'TypeError' && err.message.includes('Failed to fetch')) {
      f.error = 'Network error - check your connection and server status';
    } else {
      f.error = err.message || 'Unknown error occurred';
    }
    
    console.error('Summarize error:', err);
  }
  
  render();
}

/* Summarize all with concurrency control */
async function summarizeAll() {
  const btnAll = document.getElementById('btn-all');
  btnAll.disabled = true;
  
  const idleFiles = files.filter(f => f.status === "idle");
  
  for (const f of idleFiles) {
    await summarize(f.id);
    // Small delay between files
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
  
  btnAll.disabled = false;
}

/* Copy text */
function copyText(id, field){
  const f = files.find(x => x.id === id);
  const txt = field === "transcript" ? f.transcript : f.summary;
  if(!txt) return;
  
  navigator.clipboard.writeText(txt).then(() => {
    const btn = document.getElementById("cp-" + field + "-" + id);
    if(btn) {
      const originalText = btn.textContent;
      btn.textContent = "✓ Copied!";
      setTimeout(() => btn.textContent = originalText, 2000);
    }
  });
}

/* Render */
function render(){
  const dashboard = document.getElementById("dashboard-view");
  const history = document.getElementById("history-view");
  const showHistory = currentView === "history";
  dashboard.style.display = showHistory ? "none" : "block";
  history.style.display = showHistory ? "block" : "none";
  if(showHistory){
    renderHistory();
    return;
  }

  const total = files.length;
  const done = files.filter(f => f.status === "done").length;
  const loading = files.filter(f => f.status === "loading").length;
  const idle = files.filter(f => f.status === "idle").length;

  // Update stats
  document.getElementById("stats-row").style.display = total > 0 ? "flex" : "none";
  ["total","done","proc","idle"].forEach((k,i) => 
    document.getElementById("s-" + k).textContent = [total, done, loading, idle][i]);
  
  // Update summarize all button
  const btnAll = document.getElementById("btn-all");
  if (idle > 0 || loading > 0) {
    btnAll.style.display = "inline-flex";
    if (loading > 0) {
      document.getElementById("btn-all-lbl").textContent = `Processing... (${loading})`;
      btnAll.disabled = true;
    } else {
      document.getElementById("btn-all-lbl").textContent = `Summarize All (${idle})`;
      btnAll.disabled = false;
    }
    btnAll.onclick = summarizeAll;
  } else {
    btnAll.style.display = "none";
  }
  
  // Show/hide sections
  document.getElementById("fl-sec").style.display = total > 0 ? "block" : "none";
  document.getElementById("empty").style.display = total === 0 ? "block" : "none";
  document.getElementById("fl-count").textContent = `(${total})`;
  
  // Render cards
  document.getElementById("cards").innerHTML = files.map(renderCard).join("");
}

/* Render card */
function renderCard(f){
  const ti = typeInfo(f);
  const isAV = ti.kind === "audio" || ti.kind === "video";
  const isDoc = ti.kind === "doc";
  const isRag = isDoc && f.doc_id;
  const exp = expandedId === f.id;

  // Status pill
  const pillCls = f.status === "idle" ? "p-idle" : 
                  f.status === "loading" ? "p-load" : 
                  f.status === "done" ? "p-done" : "p-err";
  
  let pillTxt = f.status === "idle" ? "Pending" :
                f.status === "loading" ? (f.error && f.error.includes('Retrying') ? '⟳ ' + f.error : 
                  (isAV ? "● Transcribing..." : isDoc ? "● RAG Indexing..." : "● Analyzing...")) :
                f.status === "done" ? "✓ Completed" : "✕ Failed";

  // Action buttons
  let acts = "";
  if (f.status === "idle") {
    acts += `<button class="btn btn-primary" style="padding:6px 14px;font-size:12px" onclick="summarize('${f.id}')">
      ${isAV ? "🎙 Transcribe" : isDoc ? "🔍 Index & Summarize" : "Summarize"}</button>`;
  }
  if (f.status === "done") {
    acts += `<button class="btn-sm" onclick="toggleExpand('${f.id}')">${exp ? "Hide" : "View Results"}</button>`;
  }

  // Steps for audio/video
  let steps = "";
  if (isAV) {
    if (f.status === "loading") {
      steps = `<div class="steps">
        <div class="step active"><div class="step-dot"></div>Uploading</div><div class="step-arr">›</div>
        <div class="step active"><div class="step-dot"></div>Transcribing</div><div class="step-arr">›</div>
        <div class="step"><div class="step-dot"></div>Summarizing</div></div>`;
    }
    if (f.status === "done") {
      steps = `<div class="steps">
        <div class="step done"><div class="step-dot"></div>Uploaded</div><div class="step-arr">›</div>
        <div class="step done"><div class="step-dot"></div>Transcribed</div><div class="step-arr">›</div>
        <div class="step done"><div class="step-dot"></div>Summarized</div></div>`;
    }
  }

  // Steps for documents
  if (isDoc) {
    if (f.status === "loading") {
      steps = `<div class="steps">
        <div class="step active"><div class="step-dot"></div>Extracting Text</div><div class="step-arr">›</div>
        <div class="step active"><div class="step-dot"></div>Chunking & Embedding</div><div class="step-arr">›</div>
        <div class="step active"><div class="step-dot"></div>RAG Summarize</div></div>`;
    }
    if (f.status === "done" && isRag) {
      steps = `<div class="steps">
        <div class="step done"><div class="step-dot"></div>Text Extracted</div><div class="step-arr">›</div>
        <div class="step done"><div class="step-dot"></div>${f.chunk_count} Chunks Indexed</div><div class="step-arr">›</div>
        <div class="step done"><div class="step-dot"></div>RAG Summary Ready</div></div>`;
    }
  }

  // Card body
  let body = "";
  if (f.status === "loading") {
    body = `<div class="card-body">${steps}
      <div class="prog-wrap">
        <div class="prog-lbl">${isDoc ? "Chunking, embedding, and RAG indexing with Gemini..." : isAV ? "Uploading & transcribing..." : "Analyzing with Gemini AI..."}</div>
        <div class="prog-bar"><div class="prog-fill"></div></div>
        ${f.error ? `<div class="timeout-warning">⏳ ${f.error}</div>` : ''}
      </div></div>`;
  } else if (f.status === "done" && exp) {
    const hasTx = !!f.transcript;
    const at = f.activeTab || "summary";

    // Tab buttons
    let tabBtns = `<div class="tab ${at === "summary" ? "active" : ""}" onclick="setTab('${f.id}','summary')">📝 Summary</div>`;
    if (hasTx) {
      tabBtns += `<div class="tab ${at === "transcript" ? "active" : ""}" onclick="setTab('${f.id}','transcript')">🎙 Transcript</div>`;
    }

    // Summary panel
    const ragBadge = isRag ? `<div class="rag-badge">
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
      RAG-powered · ${f.chunk_count} chunks · top-8 retrieved
    </div>` : "";

    const summaryPanel = `<div class="tab-panel ${at === "summary" ? "active" : ""}">
      ${ragBadge}
      <div class="p-hdr">
        <div class="p-lbl"><div class="p-icon">✏️</div>AI Summary</div>
        <button class="btn-sm" id="cp-summary-${f.id}" onclick="copyText('${f.id}','summary')">Copy</button>
      </div>
      <div class="content-box">${esc(f.summary)}</div>
    </div>`;

    // Transcript panel
    const txPanel = hasTx ? `<div class="tab-panel ${at === "transcript" ? "active" : ""}">
      <div class="p-hdr">
        <div class="p-lbl"><div class="p-icon">✏️</div>Full Transcript</div>
        <button class="btn-sm" id="cp-transcript-${f.id}" onclick="copyText('${f.id}','transcript')">Copy</button>
      </div>
      <div class="content-box">${esc(f.transcript)}</div>
    </div>` : "";

    body = `<div class="card-body">${steps}
      <div class="tabs">${tabBtns}</div>
      ${summaryPanel}${txPanel}
    </div>`;
  } else if (f.status === "error") {
    body = `<div class="card-body"><div class="err-area">⚠ <span>${esc(f.error)}</span></div></div>`;
  }

  // Meta suffix
  const metaSuffix = isRag && f.chunk_count
    ? ` · <strong style="color:#059669">✦ RAG indexed (${f.chunk_count} chunks)</strong>`
    : isAV ? ` · <strong style="color:#D97706">Transcript + Summary</strong>` : "";

  return `<div class="file-card" id="card-${f.id}">
    <div class="file-hdr">
      <div class="type-badge" style="background:${ti.color}">${ti.badge}</div>
      <div class="file-info">
        <div class="file-name" title="${esc(f.name)}">${esc(f.name)}</div>
        <div class="file-meta">${ti.label} · ${fmtSz(f.size)}${metaSuffix}</div>
      </div>
      <span class="pill ${pillCls}">${pillTxt}</span>
      <div class="file-acts">
        ${acts}
        <button class="btn-del" onclick="removeFile('${f.id}')" title="Remove">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14H6L5 6"/>
            <path d="M10 11v6"/><path d="M14 11v6"/><path d="M9 6V4h6v2"/>
          </svg>
        </button>
      </div>
    </div>${body}</div>`;
}

// Initial render
render();
</script>
</body>
</html>"""

# ─────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not GEMINI_API_KEY:
        print("\n" + "="*70)
        print("🔴 CRITICAL: GEMINI_API_KEY not set!")
        print("\nCreate a .env file with:")
        print('  GEMINI_API_KEY="your_key_here"')
        print("="*70 + "\n")
        exit(1)
    
    if not model:
        print("\n" + "="*70)
        print("🔴 CRITICAL: Gemini model failed to initialize!")
        print("\nCheck your API key is valid and you have access to Gemini API.")
        print("="*70 + "\n")
        exit(1)
    
    print("\n" + "="*70)
    print("✅ Configuration successful!")
    print(f"📊 Using model: {model.model_name if hasattr(model, 'model_name') else 'Unknown'}")
    print(f"🔗 Embedding model: {EMBED_MODEL}")
    print("\n🚀 Starting server on http://127.0.0.1:8001")
    print("📝 Press Ctrl+C to stop")
    print("="*70 + "\n")
    
    uvicorn.run(
        "summary:app",  # Change this to match your filename (app.py)
        host="localhost", 
        port=8001, 
        reload=True,
        log_level="info"
    )
