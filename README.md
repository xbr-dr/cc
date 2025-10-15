# CampusGPT Chatbot

CampusGPT is an interactive chatbot web application designed to assist users with campus-related queries and navigation. It features an admin panel for managing documents and locations, and a user interface with chat and interactive map navigation.

---

## Features

- **User Interface**
  - Conversational chatbot powered by a Retrieval-Augmented Generation (RAG) backend
  - Interactive map with campus locations, markers, and Google Maps navigation links
  - Responsive tabs for switching between chat and map navigation
  - Leaflet-based map with satellite, terrain, and street view layers
  - Dynamic location details display

- **Admin Panel**
  - Upload and manage location data via CSV files
  - Upload and manage documents (PDF, TXT, DOCX, XLS, XLSX) for chatbot knowledge base
  - Reset location and document data
  - Export user session analytics as CSV
  - Files stored in designated knowledge base folders

- **Backend**
  - Flask REST API with CORS enabled
  - FAISS vector search index for document retrieval
  - NLTK integration for text processing
  - Modular design with blueprints for user and admin routes

---

## Project Structure

CampusGPT/
├── app.py
├── admin_routes.py
├── user_routes.py
├── rag_retriever.py
├── rag_generator.py
├── knowledge_base/
│ ├── locations/
│ └── docs/
├── user/
│ ├── index.html
│ ├── styles.css
│ └── scripts.js
├── admin/
│ ├── index.html
│ ├── styles.css
│ └── scripts.js
├── requirements.txt
└── README.md


---

## Setup and Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

