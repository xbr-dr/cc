"""
Main Flask server for CampusGPT
--------------------------------
- Registers user and admin blueprints
- Loads or rebuilds RAG index at startup
- Serves frontend templates and static files
"""

import os
from flask import Flask, render_template
from flask_cors import CORS
from admin_routes import admin_bp
from user_routes import user_bp
from rag_retriever import load_documents_and_build_index, load_index

# --- Flask App Setup ---
app = Flask(
    __name__,
    static_folder="static",     # CSS, JS files
    template_folder="templates" # HTML templates
)
CORS(app)


# --- Blueprints ---
app.register_blueprint(user_bp, url_prefix="/user")
app.register_blueprint(admin_bp, url_prefix="/admin")


# --- Home Route ---
@app.route("/")
def home():
    return render_template("user/index.html")


# --- Startup: Build or Load RAG Index ---
def initialize_rag():
    """
    Initializes the RAG index when the app starts.
    - If index files exist, loads them
    - Otherwise, builds a fresh one from documents
    """
    print("\n🚀 Initializing RAG knowledge base...")
    try:
        index_folder = "knowledge_base/index"
        if os.path.exists(index_folder) and len(os.listdir(index_folder)) > 0:
            load_index()
        else:
            load_documents_and_build_index()
        print("✅ RAG initialization complete.\n")
    except Exception as e:
        print(f"❌ Error during RAG initialization: {e}\n")


# --- Entry Point ---
if __name__ == "__main__":
    # Ensure required folders exist
    os.makedirs("knowledge_base/docs", exist_ok=True)
    os.makedirs("knowledge_base/locations", exist_ok=True)
    os.makedirs("knowledge_base/index", exist_ok=True)

    # Initialize RAG system
    initialize_rag()

    # Start Flask server
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
