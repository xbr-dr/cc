from flask import Blueprint, request, jsonify
from rag_generator import generate_answer
from admin_routes import LOCATIONS  # shared location data

user_bp = Blueprint("user_bp", __name__)

@user_bp.route("/locations", methods=["GET"])
def get_locations():
    """
    Returns the list of uploaded locations.
    """
    return jsonify(LOCATIONS)

@user_bp.route("/chat", methods=["POST"])
def chat():
    """
    Accepts chat history as JSON and returns AI-generated reply.
    Expected JSON: { "history": [{"role": "user", "message": "..."}, ...] }
    """
    data = request.get_json()
    history = data.get("history", [])

    if not history or not isinstance(history, list):
        return jsonify({"reply": "Please send a valid chat history."})

    # Call your RAG generator function
    reply = generate_answer(history)
    return jsonify({"reply": reply})
