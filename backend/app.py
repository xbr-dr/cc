"""
Main Flask Server for CampusGPT
--------------------------------
- Registers user and admin blueprints
- Loads lightweight RAG index at startup
- Serves frontend templates and static files
"""

import os
from flask import Flask, render_template
from flask_cors import CORS
from admin_routes import admin_bp
from user_routes import user_bp
from rag_retriever import load_documents_and_build_index, load_index

# Initialize Flask app
app = Flask(
    __name__,
    static_folder="static",     # CSS/JS files
    template_folder="templates" # HTML templates
)
CORS(app)

# Register blueprints
app.register_blueprint(user_bp, url_prefix="/user")
app.register_blueprint(admin_bp, url_prefix="/admin")

# Root route
@app.route("/")
def home():
    return render_template("user/index.html")

if __name__ == "__main__":
    # Load prebuilt index from disk if available, otherwise build from uploaded docs
    if os.path.exists("knowledge_base/index"):
        load_index()
    else:
        load_documents_and_build_index()

    # Start Flask server
    port = int(os.environ.get("PORT", 5000))  # for Render deployment
    app.run(host="0.0.0.0", port=port, debug=True)
