import os
import re
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
    """Extract plain text from .txt files (fallback source)."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return ""


def load_documents_and_build_index(doc_folder="knowledge_base/docs"):
    """Load documents from folder, split into chunks, and build vector index."""
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

    print(f"Loaded {len(corpus)} text chunks into memory index.")


def cosine_similarity(a, b):
    """Compute cosine similarity between vectors."""
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(a_norm, b_norm.T)


def retrieve_relevant_chunks(query, top_k=5):
    """Return top_k most similar text chunks."""
    global corpus, corpus_embeddings
    if corpus_embeddings is None or len(corpus) == 0:
        return []

    query_vec = np.array(list(embed_model.embed([query]))).astype("float32")
    sims = cosine_similarity(query_vec, corpus_embeddings)[0]
    top_indices = np.argsort(-sims)[:top_k]
    return [corpus[i] for i in top_indices]


def clear_index():
    """Reset index and corpus."""
    global corpus, corpus_embeddings
    corpus = []
    corpus_embeddings = None
