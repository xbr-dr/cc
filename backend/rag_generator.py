"""
Lightweight RAG Generator Module
--------------------------------
Uses Hugging Face Inference API (Novita provider) with Llama-3.2-1B-Instruct.
Fully serverless, no local model required.
"""

import os
import re
from rag_retriever import retrieve_relevant_chunks

try:
    from huggingface_hub import InferenceClient
except ImportError:
    raise ImportError("Please install huggingface_hub: pip install huggingface_hub")

# === CONFIGURATION ===
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

# Initialize InferenceClient with Novita provider
client = InferenceClient(
    provider="novita",
    api_key=HF_TOKEN,
)


def generate_answer(history):
    """Generates a context-aware answer from chat history using serverless Llama-3.2-1B-Instruct."""
    if not history or not isinstance(history, list):
        return "Invalid chat history."

    # Latest user message
    user_message = history[-1]["content"] if history[-1]["role"] == "user" else ""
    if not user_message.strip():
        return "Please ask a valid question."

    # Retrieve relevant knowledge chunks
    relevant_docs = retrieve_relevant_chunks(user_message, top_k=5)
    context_text = "\n\n".join(relevant_docs) if relevant_docs else "No context found."

    # System instructions
    system_instructions = """
You are CampusGPT, a helpful assistant for a college campus.
Answer ONLY questions related to the campus, its facilities, staff, departments, courses, events, contact details, and other official information.
If the question is unrelated to campus topics, politely decline to answer.
"""

    # Build messages payload
    messages = [
        {"role": "system", "content": system_instructions.strip()},
        {"role": "system", "content": f"Relevant context from documents:\n{context_text}"},
        {"role": "user", "content": user_message}
    ]

    try:
        # Serverless Novita call
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.6,
            max_tokens=250
        )

        assistant_reply = completion.choices[0].message["content"]

        # Clean output
        assistant_reply = re.sub(r"<think>.*?</think>", "", assistant_reply, flags=re.DOTALL).strip()

    except Exception as e:
        assistant_reply = f"Error generating answer: {e}"

    return assistant_reply
