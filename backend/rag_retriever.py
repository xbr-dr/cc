import os
import re
import json
import numpy as np
from fastembed import TextEmbedding

HF_TOKEN = os.getenv("HF_TOKEN")  # Render environment variable

# Initialize lightweight embedding model
if HF_TOKEN:
    embed_model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", use_auth_token=HF_TOKEN)
else:
    embed_model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

corpus = []
corpus_embeddings = None


def simple_sentence_split(text):
    """Lightweight sentence splitter using regex (no nltk)."""
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


def load_documents_and_build_index(doc_folder="knowledge_base/docs"):
    """Load documents, embed them, and save the index persistently."""
    global corpus, corpus_embeddings

    if not os.path.exists(doc_folder):
        print(f"Document folder '{doc_folder}' does not exist. Skipping index build.")
        corpus, corpus_embeddings = [], None
        return

    all_text = ""
    for filename in os.listdir(doc_folder):
        filepath = os.path.join(doc_folder, filename)
        if filename.lower().endswith(".txt"):
            text = extract_text_from_txt(filepath)
            all_text += text + "\n"
        else:
            print(f"Skipping unsupported file type: {filename}")

    if not all_text.strip():
        print("No text extracted from documents.")
        corpus, corpus_embeddings = [], None
        return

    corpus = simple_sentence_split(all_text)
    corpus_embeddings = np.array(list(embed_model.embed(corpus))).astype("float32")

    os.makedirs("knowledge_base/index", exist_ok=True)
    np.save("knowledge_base/index/corpus_embeddings.npy", corpus_embeddings)
    with open("knowledge_base/index/corpus.json", "w", encoding="utf-8") as f:
        json.dump(corpus, f)

    print(f"✅ Loaded and saved {len(corpus)} text chunks into memory index.")


def load_index():
    """Load prebuilt index from disk if available."""
    global corpus, corpus_embeddings
    try:
        emb_path = "knowledge_base/index/corpus_embeddings.npy"
        corpus_path = "knowledge_base/index/corpus.json"

        if os.path.exists(emb_path) and os.path.exists(corpus_path):
            corpus_embeddings = np.load(emb_path)
            with open(corpus_path, "r", encoding="utf-8") as f:
                corpus = json.load(f)
            print(f"✅ Loaded index with {len(corpus)} chunks from disk.")
        else:
            print("⚠️ No saved index found; you need to upload documents first.")
    except Exception as e:
        print(f"Error loading index: {e}")


def cosine_similarity(a, b):
    """Compute cosine similarity between vectors."""
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(a_norm, b_norm.T)


def retrieve_relevant_chunks(query, top_k=5):
    """Return top_k most similar text chunks."""
    global corpus, corpus_embeddings
    if corpus_embeddings is None or len(corpus) == 0:
        print("⚠️ No index loaded; returning empty context.")
        return []

    query_vec = np.array(list(embed_model.embed([query]))).astype("float32")
    sims = cosine_similarity(query_vec, corpus_embeddings)[0]
    top_indices = np.argsort(-sims)[:top_k]
    return [corpus[i] for i in top_indices]


def clear_index():
    """Reset index and delete stored embeddings."""
    global corpus, corpus_embeddings
    corpus = []
    corpus_embeddings = None

    try:
        folder = "knowledge_base/index"
        if os.path.exists(folder):
            for f in os.listdir(folder):
                os.remove(os.path.join(folder, f))
        print("🧹 Cleared index files.")
    except Exception as e:
        print(f"Error clearing index: {e}")
