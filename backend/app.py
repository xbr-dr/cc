from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from openai import OpenAI
from rag_retriever import (
    load_documents_and_build_index,
    load_index,
    retrieve_relevant_chunks,
    clear_index,
)

load_dotenv()

app = Flask(__name__)

UPLOAD_FOLDER = "knowledge_base/docs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize model client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load index if exists
load_index()


@app.route("/")
def home():
    return render_template("user/index.html")


@app.route("/admin/")
def admin_home():
    return render_template("admin/index.html")


@app.route("/admin/upload_documents", methods=["POST"])
def upload_documents():
    files = request.files.getlist("documents")
    for file in files:
        if file and file.filename:
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

    load_documents_and_build_index(UPLOAD_FOLDER)
    return jsonify({"message": "✅ Documents uploaded and indexed successfully."})


@app.route("/admin/reset_documents", methods=["POST"])
def reset_documents():
    clear_index()
    for f in os.listdir(UPLOAD_FOLDER):
        os.remove(os.path.join(UPLOAD_FOLDER, f))
    return jsonify({"message": "🧹 All documents and embeddings cleared."})


@app.route("/user/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "")

    context_chunks = retrieve_relevant_chunks(user_message)
    context = "\n\n".join(context_chunks) if context_chunks else "No context available."

    prompt = f"""
You are a helpful assistant. Use the context below to answer the question accurately.
If the answer is not in the context, say so politely.

Context:
{context}

Question: {user_message}
Answer:
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )

    return jsonify({"reply": response.choices[0].message.content.strip()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
