"""
Main Flask server for CampusGPT
--------------------------------
- Registers user and admin blueprints
- Loads lightweight RAG index at startup
- Serves frontend templates and static files
"""

from flask import Flask, redirect, render_template
from flask_cors import CORS
from admin_routes import admin_bp
from user_routes import user_bp
from rag_retriever import load_documents_and_build_index

app = Flask(
    __name__,
    static_folder="static",    # your CSS/JS
    template_folder="templates"  # your HTML
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
    # Load or rebuild index at startup
    load_documents_and_build_index()

    # Start Flask server
    app.run(host="0.0.0.0", port=5000, debug=True)
