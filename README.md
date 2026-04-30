# EasyHire — CV Extraction & Analysis System

> Automating CV Extraction and Analysis Using NLP and Document AI

A lightweight, AI-powered recruitment screening tool built for small and medium-sized enterprises (SMEs) in Zimbabwe. EasyHire parses CVs in multiple formats, extracts and matches skills to job descriptions, ranks candidates transparently, and performs basic certificate authenticity checks — all from a single Streamlit dashboard.

---

## Problem

SMEs in Zimbabwe typically lack dedicated HR departments. Hiring is handled by owners or administrators who receive CVs via WhatsApp, email, and paper in inconsistent formats (scanned images, PDFs, Word documents). Manual screening is slow, inconsistent, and prone to bias. Existing Applicant Tracking Systems are expensive, bandwidth-heavy, and impractical for lean teams.

## Solution

EasyHire automates the first-pass shortlisting stage of recruitment by combining document parsing, semantic search, and generative AI into a single affordable workflow that runs on ordinary hardware.

---

## Features

- **Multi-format CV parsing** — Extracts text from PDF, DOCX, and plain-text files using PyMuPDF and python-docx.
- **Semantic embeddings & vector search** — Converts CV content into vector embeddings (BGE-Small) and indexes them with FAISS for fast similarity retrieval.
- **RAG-powered analysis** — Retrieves the most relevant CV chunks and feeds them to Google Gemini for context-aware candidate evaluation.
- **Skill extraction & matching** — Identifies required skills from job descriptions, matches them against CV content, and highlights gaps.
- **Match percentage scoring** — Produces a numerical suitability score combining keyword matches, semantic relevance, and LLM scoring.
- **ATS-style candidate reports** — Generates structured feedback: profile summary, strengths, weaknesses, and missing skills.
- **Candidate ranking dashboard** — Displays ranked shortlists with filters for experience, match percentage, and detected skills.
- **Certificate verification** — Performs lightweight integrity checks on PDF and JPEG certificates using metadata inspection (pikepdf, exifread) to flag potential tampering.
- **CSV export** — Exports ranked candidate data with detected skills for offline review.
- **SQLite storage** — Persists processed CV data and results for later access.

---

## System Architecture

The system uses a **Retrieval-Augmented Generation (RAG)** architecture:

![System Architecture](https://raw.githubusercontent.com/tariwecode/Automating-CV-extraction-and-analysis-using-NLP-and-document-AI/main/diagrams/System_Architecture.jpg)

1. CVs are parsed and chunked into text segments.
2. Chunks are embedded using the BGE-Small model and stored in a FAISS vector index.
3. When a job description is provided, the system retrieves the most relevant CV chunks via semantic similarity.
4. Retrieved context + the job description are sent to Google Gemini (1.5-Flash) for analysis, skill extraction, and match scoring.
5. Results are displayed in a Streamlit dashboard and stored in SQLite.

---

## Tech Stack

| Component              | Technology                              |
| ---------------------- | --------------------------------------- |
| Language               | Python 3.x                              |
| Web framework          | Streamlit                                |
| Document parsing       | PyMuPDF, python-docx                     |
| Embeddings             | FastEmbed (BAAI/bge-small-en-v1.5)       |
| Vector store           | FAISS                                    |
| Generative AI          | Google Generative AI (Gemini 1.5-Flash)  |
| Database               | SQLite                                   |
| Certificate checks     | pikepdf, exifread, Pillow                |
| Data handling          | pandas                                   |

---

## Project Structure

```
.
├── app.py                  # Main Streamlit application
├── app_deploy.py           # Deployment configuration
├── auth.py                 # Authentication module
├── dashboard.py            # Dashboard interface logic
├── evaluation.py           # System evaluation scripts
├── .env                    # Environment variables (API keys)
├── .gitignore
├── requirements.txt        # Python dependencies
├── requirements copy.txt
├── CV_data                 # SQLite database file
├── CV_data1                # Secondary database file
├── resume_data             # Resume data store
├── users                   # User data store
├── evaluation_results      # Evaluation metrics output
├── chroma_store1/          # Vector store directory
├── chroma_store2/          # Secondary vector store
├── vector_store/           # FAISS vector index
├── embeddings/             # Embedding model cache
├── datasets/               # CV and job description datasets
├── temp_files/             # Temporary processing files
├── temp_files1/            # Secondary temp directory
├── temp_uploads/           # Uploaded file staging
└── .venv/                  # Python virtual environment
```

---

## Setup & Installation

### Prerequisites

- Python 3.8+
- A Google Generative AI API key (Gemini)

### Steps

```bash
# Clone the repository
git clone <repo-url>
cd <project-folder>

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
# Create a .env file in the project root:
echo "GOOGLE_API_KEY=your_api_key_here" > .env

# Run the application
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## Usage

1. **Add a job description** — Paste or type the role requirements in the sidebar.
2. **Upload CVs** — Drag and drop PDF, DOCX, or TXT files (up to 200 MB each).
3. **Submit** — The system parses, embeds, and indexes all uploaded CVs.
4. **Individual Analysis** — Select a CV to view its profile summary, strengths, weaknesses, missing skills, and match percentage.
5. **Dashboard** — Switch to the Dashboard tab for ranked candidate lists, filters, summary statistics, and full CV overview.
6. **Certificate Verification** — Expand the verification section to upload a PDF or JPEG certificate and view the authenticity verdict.
7. **Export** — Download ranked results as a CSV file.

---

## Methodology

The project follows a **Design Science Research (DSR)** approach combined with **Agile development** practices.

![Design Research Diagram](https://raw.githubusercontent.com/tariwecode/Automating-CV-extraction-and-analysis-using-NLP-and-document-AI/main/diagrams/Design_Research_Diagram.jpg)

![Agile Development](https://raw.githubusercontent.com/tariwecode/Automating-CV-extraction-and-analysis-using-NLP-and-document-AI/main/diagrams/Agile_Dev_Diagram.jpg)

### Evaluation

The system was tested with 10 real CVs and evaluated by 5 SME business owners:

- **Speed** — 10 CVs processed in ~40 s (fast network) to ~78 s (slow network), compared to days/weeks manually.
- **User satisfaction** — 100% of testers confirmed faster screening, clearer rankings, and reduced bias.
- **Certificate verification** — Correctly identified 2 genuine certificates and flagged 1 tampered document.

---

## Limitations

- Prototype-stage — tested with a small sample (10 CVs, 5 users).
- Requires internet connectivity for Gemini API calls.
- Certificate verification is metadata-based only (no cryptographic validation or deep forensics).
- Extraction accuracy depends on CV formatting quality.
- Subject to Gemini API quota limits during heavy use.

---

## Future Work

- Advanced certificate forgery and deepfake detection.
- WhatsApp/SMS-based CV submission integration.
- Hybrid architecture (local models + cloud LLMs) to reduce API dependency.
- Employer accounts and vacancy management features.
- Training on local Zimbabwean CV datasets for improved extraction accuracy.

---

## Author

**Tariro Coffee**

---

## License

This project was developed as an academic capstone. Contact the author for usage and licensing inquiries.
