# rag_retriever.py
import os
import re
import json
import numpy as np
import pandas as pd
from fastembed import TextEmbedding
import fitz  # PyMuPDF for PDF parsing

# -----------------------------
# Initialization
# -----------------------------
HF_TOKEN = os.getenv("HF_TOKEN")  # Hugging Face auth (optional)

if HF_TOKEN:
    embed_model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", use_auth_token=HF_TOKEN)
else:
    embed_model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

corpus = []
corpus_metadata = []
corpus_embeddings = None

# -----------------------------
# Utility regexes & constants
# -----------------------------
EMAIL_RE = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')
PHONE_RE = re.compile(r'(?:\+?\d{1,3}[-.\s]?)?(?:\(\d{2,4}\)|\d{2,4})[-.\s]?\d{5,12}')
LABEL_RE = re.compile(r'^(?:full name|name|department|position|qualification|email|mobile|mobile number|area of specialization)\s*[:\-]', re.I)
BULLET_RE = re.compile(r'^[\u2022\-\*\•\d\)\.]+\s+', re.M)
HEADING_RE = re.compile(r'^(?:#{1,6}\s*|[A-Z][A-Z\s]{3,}\s*$|[A-Z][a-z]+\s*[-]{2,}|^[A-Z][\w\s]{10,}$)', re.M)
MIN_CHUNK_CHARS = 60
MAX_CHUNK_CHARS = 1200

# -----------------------------
# Text extraction
# -----------------------------
def extract_text_from_txt(filepath):
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return [(1, f.read())]
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return []

def extract_text_from_pdf(filepath):
    pages = []
    try:
        with fitz.open(filepath) as pdf:
            for i, page in enumerate(pdf):
                pages.append((i + 1, page.get_text("text")))
    except Exception as e:
        print(f"Error reading PDF {filepath}: {e}")
    return pages

def extract_text_from_csv(filepath):
    pages = []
    try:
        df = pd.read_csv(filepath, dtype=str, keep_default_na=False)
        for idx, row in df.iterrows():
            parts = [f"{col}: {str(row[col]).strip()}" for col in df.columns if str(row[col]).strip()]
            text = " | ".join(parts)
            if text.strip():
                pages.append((idx + 1, text))
    except Exception as e:
        print(f"Error reading CSV {filepath}: {e}")
    return pages

def extract_text(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(filepath)
    elif ext == ".txt":
        return extract_text_from_txt(filepath)
    elif ext == ".csv":
        return extract_text_from_csv(filepath)
    else:
        print(f"⚠️ Unsupported file type: {filepath}")
        return []

# -----------------------------
# Smart Dynamic Chunking
# -----------------------------
def normalize_whitespace(s):
    return re.sub(r'\s+', ' ', s).strip()

def split_paragraphs(text):
    parts = re.split(r'\n\s*\n+', text)
    return [p.strip() for p in parts if len(p.strip()) > 30]

def split_structured_records(text):
    if "faculty member" in text.lower():
        parts = re.split(r'faculty member\s*\d*\s*', text, flags=re.I)
        return [p.strip() for p in parts if len(p.strip()) > MIN_CHUNK_CHARS]
    if text.lower().count("full name") >= 2:
        parts = re.split(r'(?i)(?:\n|^)\s*full name\s*[:\-]\s*', text)
        if parts and len(parts[0].strip()) < 20:
            parts = parts[1:]
        cleaned = []
        for p in parts:
            if len(p.strip()) >= MIN_CHUNK_CHARS:
                cleaned.append("Full Name: " + p.strip())
        return cleaned
    return []

def split_bulleted_lists(text):
    lines = text.splitlines()
    out, buf = [], []
    for l in lines:
        if BULLET_RE.match(l):
            buf.append(BULLET_RE.sub("", l).strip())
        else:
            if buf:
                out.append(" ".join(buf))
                buf = []
            out.append(l)
    if buf:
        out.append(" ".join(buf))
    return [o.strip() for o in out if len(o.strip()) > 30]

def split_by_headings(text):
    lines = text.splitlines()
    chunks, buffer = [], []
    for line in lines:
        if HEADING_RE.match(line.strip()) and buffer:
            chunks.append("\n".join(buffer).strip())
            buffer = [line]
        else:
            buffer.append(line)
    if buffer:
        chunks.append("\n".join(buffer).strip())

    out = []
    for c in chunks:
        if len(c) > MAX_CHUNK_CHARS:
            out.extend(split_paragraphs(c))
        else:
            out.append(c)
    return [o for o in out if len(o) > 30]

def fallback_sentence_split(text):
    sents = re.split(r'(?<=[.!?])\s+', text)
    sents = [s.strip() for s in sents if s.strip()]
    chunks, current = [], ""
    for s in sents:
        if len(current) + len(s) + 1 <= MAX_CHUNK_CHARS:
            current = (current + " " + s).strip()
        else:
            if len(current) >= MIN_CHUNK_CHARS:
                chunks.append(current)
            current = s
    if current and len(current) >= MIN_CHUNK_CHARS:
        chunks.append(current)
    return chunks if chunks else [text.strip()[:MAX_CHUNK_CHARS]]

def smart_chunk_text(text):
    text = text.strip()
    if len(text) < MIN_CHUNK_CHARS:
        return []
    records = split_structured_records(text)
    if records:
        return [normalize_whitespace(r) for r in records]
    if BULLET_RE.search(text):
        return [normalize_whitespace(c) for c in split_bulleted_lists(text)]
    if HEADING_RE.search(text):
        return [normalize_whitespace(c) for c in split_by_headings(text)]
    paras = split_paragraphs(text)
    if len(paras) > 1:
        chunks = []
        for p in paras:
            if len(p) > MAX_CHUNK_CHARS:
                chunks.extend(fallback_sentence_split(p))
            else:
                chunks.append(p)
        return [normalize_whitespace(c) for c in chunks]
    return [normalize_whitespace(c) for c in fallback_sentence_split(text)]

# -----------------------------
# Index building and loading
# -----------------------------
def load_documents_and_build_index(doc_folder="knowledge_base/docs"):
    """Dynamic, general-purpose index builder."""
    global corpus, corpus_metadata, corpus_embeddings

    if not os.path.exists(doc_folder):
        print(f"⚠️ Folder '{doc_folder}' does not exist.")
        corpus, corpus_metadata, corpus_embeddings = [], [], None
        return

    corpus_data = []
    for filename in os.listdir(doc_folder):
        filepath = os.path.join(doc_folder, filename)
        if not os.path.isfile(filepath):
            continue
        print(f"📄 Processing: {filename}")
        pages = extract_text(filepath)
        for page_num, text in pages:
            if not text or len(text.strip()) < 30:
                continue
            chunks = smart_chunk_text(text)
            for i, chunk in enumerate(chunks):
                meta = {
                    "source_file": filename,
                    "page": page_num,
                    "chunk_id": f"{page_num}_{i}",
                    "length": len(chunk),
                    "contains_email": bool(EMAIL_RE.search(chunk)),
                    "contains_phone": bool(PHONE_RE.search(chunk)),
                }
                corpus_data.append({"text": chunk, "meta": meta})

    if not corpus_data:
        print("⚠️ No text extracted from documents.")
        corpus, corpus_metadata, corpus_embeddings = [], [], None
        return

    corpus = [c["text"] for c in corpus_data]
    corpus_metadata = [c["meta"] for c in corpus_data]
    embeddings = []
    batch_size = 64
    for i in range(0, len(corpus), batch_size):
        batch = corpus[i:i+batch_size]
        embeddings.extend(list(embed_model.embed(batch)))

    corpus_embeddings = np.array(embeddings).astype("float32")

    os.makedirs("knowledge_base/index", exist_ok=True)
    np.save("knowledge_base/index/corpus_embeddings.npy", corpus_embeddings)
    with open("knowledge_base/index/corpus.json", "w", encoding="utf-8") as f:
        json.dump(corpus_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Built dynamic index with {len(corpus_data)} chunks.")

def load_index():
    """Load saved dynamic index if available."""
    global corpus, corpus_metadata, corpus_embeddings
    try:
        emb_path = "knowledge_base/index/corpus_embeddings.npy"
        corpus_path = "knowledge_base/index/corpus.json"
        if os.path.exists(emb_path) and os.path.exists(corpus_path):
            corpus_embeddings = np.load(emb_path)
            with open(corpus_path, "r", encoding="utf-8") as f:
                corpus_data = json.load(f)
            corpus = [c["text"] for c in corpus_data]
            corpus_metadata = [c["meta"] for c in corpus_data]
            print(f"✅ Loaded {len(corpus)} chunks from saved index.")
        else:
            print("⚠️ No saved index found — upload documents first.")
    except Exception as e:
        print(f"Error loading index: {e}")

# -----------------------------
# Retrieval
# -----------------------------
def cosine_similarity(a, b):
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(a_norm, b_norm.T)

def retrieve_relevant_chunks(query, top_k=5):
    """Retrieve top_k relevant chunks with metadata-aware boosting."""
    global corpus, corpus_metadata, corpus_embeddings
    if corpus_embeddings is None or len(corpus) == 0:
        print("⚠️ No index loaded; returning empty context.")
        return []

    # Preprocess query for better matching
    query_lower = query.lower()
    
    # Detect if this is a contact information query
    is_contact_query = any(word in query_lower for word in 
                          ["email", "contact", "phone", "mobile", "number", "call", "reach", "ph"])
    
    # Generate query embedding
    query_vec = np.array(list(embed_model.embed([query]))).astype("float32")
    sims = cosine_similarity(query_vec, corpus_embeddings)[0]
    
    # Apply metadata-based boosting for contact queries
    if is_contact_query and corpus_metadata:
        for i, meta in enumerate(corpus_metadata):
            # Boost chunks that contain email/phone when user asks for contact info
            if meta.get("contains_email") or meta.get("contains_phone"):
                sims[i] *= 1.3  # 30% boost for chunks with contact info
    
    # Get top results
    top_indices = np.argsort(-sims)[:top_k]

    results = []
    for i in top_indices:
        results.append({
            "text": corpus[i],
            "similarity": float(sims[i])
        })
    return results

# -----------------------------
# Maintenance
# -----------------------------
def clear_index():
    """Clear embeddings and index data."""
    global corpus, corpus_metadata, corpus_embeddings
    corpus, corpus_metadata, corpus_embeddings = [], [], None
    try:
        folder = "knowledge_base/index"
        if os.path.exists(folder):
            for f in os.listdir(folder):
                os.remove(os.path.join(folder, f))
        print("🧹 Cleared index.")
    except Exception as e:
        print(f"Error clearing index: {e}")
