from flask import Blueprint, request, jsonify, render_template
import os
import csv
from werkzeug.utils import secure_filename

admin_bp = Blueprint("admin_bp", __name__)

# Shared data
LOCATIONS = []
DOCUMENTS = []

def save_files(files, folder):
    os.makedirs(folder, exist_ok=True)
    saved_files = []
    for f in files:
        filename = secure_filename(f.filename)
        filepath = os.path.join(folder, filename)
        f.save(filepath)
        saved_files.append(filepath)
    return saved_files

def load_locations_from_disk(folder="knowledge_base/locations"):
    """Load all location CSV files from disk into LOCATIONS list."""
    global LOCATIONS
    LOCATIONS.clear()
    
    if not os.path.exists(folder):
        print(f"⚠️ Locations folder '{folder}' does not exist.")
        return
    
    existing_names = set()
    loaded_count = 0
    
    for filename in os.listdir(folder):
        if not filename.lower().endswith('.csv'):
            continue
            
        filepath = os.path.join(folder, filename)
        print(f"📍 Loading locations from: {filename}")
        
        try:
            with open(filepath, newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    try:
                        name = row["name"].strip()
                        # Prevent duplicates
                        if name.lower() not in existing_names:
                            loc = {
                                "name": name,
                                "details": row.get("details", "").strip(),
                                "lat": float(row["lat"]),
                                "lon": float(row["lon"])
                            }
                            LOCATIONS.append(loc)
                            existing_names.add(name.lower())
                            loaded_count += 1
                    except (KeyError, ValueError) as e:
                        print(f"⚠️ Skipping invalid row in {filename}: {row} - Error: {e}")
        except Exception as e:
            print(f"❌ Error reading {filepath}: {e}")
    
    print(f"✅ Loaded {loaded_count} locations from disk.")

@admin_bp.route("/", methods=["GET"])
def admin_home():
    return render_template("admin/index.html")

@admin_bp.route("/upload_locations", methods=["POST"])
def upload_locations():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"status": "error", "message": "No files uploaded"}), 400
    
    # Save files to disk
    saved_files = save_files(files, "knowledge_base/locations")
    
    # Reload all locations from disk (prevents duplicates and ensures consistency)
    load_locations_from_disk()
    
    return jsonify({
        "status": "success",
        "files_uploaded": len(files),
        "total_locations": len(LOCATIONS)
    })

@admin_bp.route("/upload_documents", methods=["POST"])
def upload_documents():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"status": "error", "message": "No files uploaded"}), 400

    saved_files = save_files(files, "knowledge_base/docs")
    DOCUMENTS.extend(saved_files)

    # Import here to avoid circular imports
    from rag_retriever import load_documents_and_build_index
    load_documents_and_build_index()

    return jsonify({
        "status": "success",
        "files_uploaded": len(files),
        "docs_added": len(saved_files)
    })

@admin_bp.route("/reset_locations", methods=["POST"])
def reset_locations():
    LOCATIONS.clear()
    loc_folder = "knowledge_base/locations"
    if os.path.exists(loc_folder):
        for f in os.listdir(loc_folder):
            os.remove(os.path.join(loc_folder, f))
    return jsonify({"status": "success", "message": "All locations reset."})

@admin_bp.route("/reset_documents", methods=["POST"])
def reset_documents():
    DOCUMENTS.clear()
    doc_folder = "knowledge_base/docs"
    if os.path.exists(doc_folder):
        for f in os.listdir(doc_folder):
            os.remove(os.path.join(doc_folder, f))
    # Clear FAISS index and corpus in rag_retriever module
    from rag_retriever import clear_index
    clear_index()
    return jsonify({"status": "success", "message": "All documents reset."})

@admin_bp.route("/export_analytics", methods=["GET"])
def export_analytics():
    csv_content = "session_id,user,actions\n1,UserA,chat\n2,UserB,navigate\n"
    return (csv_content, 200, {
        "Content-Type": "text/csv",
        "Content-Disposition": "attachment; filename=user_session_analytics.csv"
    })