import os
import re
import json
import numpy as np
import pandas as pd
from fastembed import TextEmbedding

import fitz  # PyMuPDF for PDF parsing

HF_TOKEN = os.getenv("HF_TOKEN")  # Hugging Face auth (optional)

# Initialize lightweight embedding model
if HF_TOKEN:
    embed_model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", use_auth_token=HF_TOKEN)
else:
    embed_model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

corpus = []
corpus_embeddings = None


def simple_sentence_split(text):
    """Lightweight sentence splitter using regex."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def extract_text_from_txt(filepath):
    """Extract plain text from .txt files."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return ""


def extract_text_from_pdf(filepath):
    """Extract text from PDF using PyMuPDF."""
    text = ""
    try:
        with fitz.open(filepath) as pdf:
            for page in pdf:
                text += page.get_text("text") + "\n"
    except Exception as e:
        print(f"Error reading PDF {filepath}: {e}")
    return text


def extract_text_from_csv(filepath):
    """Extract readable text from CSV — combine headers and rows."""
    try:
        df = pd.read_csv(filepath)
        text = df.to_string(index=False)
        return text
    except Exception as e:
        print(f"Error reading CSV {filepath}: {e}")
        return ""


def extract_text(filepath):
    """General extractor that handles multiple file types."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".txt":
        return extract_text_from_txt(filepath)
    elif ext == ".pdf":
        return extract_text_from_pdf(filepath)
    elif ext == ".csv":
        return extract_text_from_csv(filepath)
    else:
        print(f"⚠️ Unsupported file type: {filepath}")
        return ""


def load_documents_and_build_index(doc_folder="knowledge_base/docs"):
    """Load docs, extract text, embed, and save index."""
    global corpus, corpus_embeddings

    if not os.path.exists(doc_folder):
        print(f"⚠️ Folder '{doc_folder}' does not exist.")
        corpus, corpus_embeddings = [], None
        return

    all_text = ""
    for filename in os.listdir(doc_folder):
        filepath = os.path.join(doc_folder, filename)
        print(f"📄 Processing: {filename}")
        text = extract_text(filepath)
        if text.strip():
            all_text += text + "\n"

    if not all_text.strip():
        print("⚠️ No text extracted from documents.")
        corpus, corpus_embeddings = [], None
        return

    corpus = simple_sentence_split(all_text)
    corpus_embeddings = np.array(list(embed_model.embed(corpus))).astype("float32")

    os.makedirs("knowledge_base/index", exist_ok=True)
    np.save("knowledge_base/index/corpus_embeddings.npy", corpus_embeddings)
    with open("knowledge_base/index/corpus.json", "w", encoding="utf-8") as f:
        json.dump(corpus, f)

    print(f"✅ Built index with {len(corpus)} chunks.")


def load_index():
    """Load saved index if available."""
    global corpus, corpus_embeddings
    try:
        emb_path = "knowledge_base/index/corpus_embeddings.npy"
        corpus_path = "knowledge_base/index/corpus.json"

        if os.path.exists(emb_path) and os.path.exists(corpus_path):
            corpus_embeddings = np.load(emb_path)
            with open(corpus_path, "r", encoding="utf-8") as f:
                corpus = json.load(f)
            print(f"✅ Loaded {len(corpus)} chunks from saved index.")
        else:
            print("⚠️ No saved index found — upload documents first.")
    except Exception as e:
        print(f"Error loading index: {e}")


def cosine_similarity(a, b):
    """Compute cosine similarity between vectors."""
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(a_norm, b_norm.T)


def retrieve_relevant_chunks(query, top_k=5):
    """Return top_k most relevant text chunks."""
    global corpus, corpus_embeddings
    if corpus_embeddings is None or len(corpus) == 0:
        print("⚠️ No index loaded; returning empty context.")
        return []

    query_vec = np.array(list(embed_model.embed([query]))).astype("float32")
    sims = cosine_similarity(query_vec, corpus_embeddings)[0]
    top_indices = np.argsort(-sims)[:top_k]
    return [corpus[i] for i in top_indices]


def clear_index():
    """Clear all embeddings and cached data."""
    global corpus, corpus_embeddings
    corpus = []
    corpus_embeddings = None
    try:
        folder = "knowledge_base/index"
        if os.path.exists(folder):
            for f in os.listdir(folder):
                os.remove(os.path.join(folder, f))
        print("🧹 Cleared index.")
    except Exception as e:
        print(f"Error clearing index: {e}")
