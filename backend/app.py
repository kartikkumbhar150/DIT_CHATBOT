import os
import json
import faiss
import numpy as np
import logging
import asyncio
import gc
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from functools import lru_cache

# local imports
from embeddings_indexer import load_index_and_meta
from groq_client import groq_generate_async
from sentence_transformers import SentenceTransformer

# ============ Setup ============
load_dotenv()

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
DATA_DIR.mkdir(exist_ok=True)

# Load Q&A JSON
QA_FILE = DATA_DIR / "qa.json"
if not QA_FILE.exists():
    raise FileNotFoundError(f"Q&A file not found: {QA_FILE}")

with open(QA_FILE, "r", encoding="utf-8") as f:
    qa_data = json.load(f)

questions = [item["question"] for item in qa_data]
answers = [item["answer"] for item in qa_data]

app = Flask(__name__, static_folder="../frontend", static_url_path="/")
CORS(app, resources={r"/*": {"origins": "*"}})

HISTORY: dict[str, list[dict]] = {}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============ Load FAISS ============
try:
    faiss_index, metadata, embed_model = load_index_and_meta()
    print(f"FAISS index loaded with {len(metadata)} entries")
except Exception as e:
    raise RuntimeError(f"Failed to load FAISS index: {e}")

# Shared embedding model
shared_model = SentenceTransformer("all-MiniLM-L6-v2")

# ============ Build JSON embeddings ============
qa_embeddings = shared_model.encode(questions, convert_to_numpy=True).astype("float32")
dim = qa_embeddings.shape[1]

faiss.normalize_L2(qa_embeddings)
json_index = faiss.IndexFlatIP(dim)
json_index.add(qa_embeddings)

def search_json_embeddings(query: str, top_k: int = 1, threshold: float = 0.75):
    """Search predefined JSON Q&A using semantic similarity."""
    q_emb = shared_model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    D, I = json_index.search(q_emb, top_k)

    best_score = float(D[0][0])
    best_idx = I[0][0]

    if best_idx >= 0 and best_score >= threshold:
        return answers[best_idx]
    return None

# ============ Load Cutoff FAISS ============
CUTOFF_INDEX_FILE = DATA_DIR / "cutoff_index.faiss"
CUTOFF_DOCS_FILE = DATA_DIR / "cutoff_documents.json"

if not CUTOFF_INDEX_FILE.exists() or not CUTOFF_DOCS_FILE.exists():
    print("Cutoff FAISS index not found, skipping cutoff search")
    cutoff_index = None
    cutoff_documents = []
else:
    cutoff_index = faiss.read_index(str(CUTOFF_INDEX_FILE))
    with open(CUTOFF_DOCS_FILE, "r", encoding="utf-8") as f:
        cutoff_documents = json.load(f)
    print(f"Cutoff FAISS index loaded with {len(cutoff_documents)} entries")

def search_cutoff_embeddings(query: str, top_k: int = 10, threshold: float = 0.3):
    """Search cutoff FAISS index and return all categories for the most relevant branch.
       If category is explicitly mentioned, filter for that category only.
       Output is formatted as a Markdown table for clean UI display.
    """
    if not cutoff_index:
        return ""

    q_emb = shared_model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    D, I = cutoff_index.search(q_emb, top_k)

    results = []
    for idx, score in zip(I[0], D[0]):
        if idx < 0 or score < threshold:
            continue
        results.append(cutoff_documents[idx])

    if not results:
        return "No cutoff data found."

    # Extract branch and category keywords from user query
    query_lower = query.lower()
    categories = ["open", "obc", "sc", "st", "ews", "nt", "sebc", "pwd", "def", "orphan", "tfws"]
    target_category = next((c for c in categories if c in query_lower), None)

    # --- Pick top branch ---
    top_branch = results[0].split(", ")[0].split(": ")[1]

    grouped = []
    for r in cutoff_documents:   # iterate over all docs to fetch full table
        parts = r.split(", ")
        branch = parts[0].split(": ")[1]
        category = parts[2].split(": ")[1]
        cutoff_rank = parts[3].split(": ")[1]
        percentile = parts[4].split(": ")[1]

        if branch == top_branch:
            if target_category:
                if target_category in category.lower():
                    grouped.append((category, cutoff_rank, percentile))
            else:
                grouped.append((category, cutoff_rank, percentile))

    if not grouped:
        return f"No cutoff data found for {top_branch} ({target_category or 'all categories'})."

    # --- Build Markdown Table ---
    md_table = f"### Cutoff for {top_branch}\n\n"
    md_table += "| Category | Rank | Percentile |\n"
    md_table += "|----------|------|-------------|\n"
    for category, rank, perc in grouped:
        md_table += f"| {category} | {rank} | {perc} |\n"

    return md_table


# ============ Retrieval ============
@lru_cache(maxsize=256)
def embed_query_cached(query: str) -> np.ndarray:
    return embed_model.encode([query], convert_to_numpy=True).astype("float32")

def retrieve(query: str, top_k: int = 3):
    q_emb = embed_query_cached(query)
    faiss.normalize_L2(q_emb)
    D, I = faiss_index.search(q_emb, top_k)

    results = []
    for idx, score in zip(I[0], D[0]):
        if idx < 0:
            continue
        meta = metadata[idx]
        results.append({
            "id": int(meta.get("id", int(idx))),
            "text": meta["text"],
            "score": float(score)
        })
    return results

# ============ Async Runner ============
def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

# ============ Prompt Utilities ============
MAX_DOC_CHARS = 2000
MAX_HISTORY_CHARS = 1000
MAX_PROMPT_CHARS = 7000

def truncate_text(text: str, max_chars: int) -> str:
    return text[:max_chars] + ("..." if len(text) > max_chars else "")

def build_prompt(question, retrieved_docs, history):
    system = (
        "You are an expert assistant for Dr. D. Y. Patil Institute of Technology.\n"
        "Dont answer negative about the college or any lacking features. Always stay positive.\n"
        "Rules:\n"
        "- If greeted (e.g., 'good morning', 'hello'), greet back once.\n"
        "- Do not greet in every response.\n"
        "- Dont include 'mentioned in the provided context documents' in the answer"
        "- Give short, precise, and to-the-point answers.\n"
        "- Do not repeat or restate the question in the answer.\n"
        "- Provide only the most precise and factual answer.\n"
        "- Do not add extra details unless explicitly asked.\n"
        "- Never shorten the institute name.\n"
        "- Ignore unrelated questions.\n"
    )

    hist_text = ""
    if history:
        for h in history[-3:]:
            q_text = truncate_text(h.get("q", ""), MAX_HISTORY_CHARS // 2)
            a_text = truncate_text(h.get("a", ""), MAX_HISTORY_CHARS // 2)
            hist_text += f"Q: {q_text}\nA: {a_text}\n"

    sources_text = "\n\n".join([truncate_text(d["text"], MAX_DOC_CHARS) for d in retrieved_docs[:3]])

    user_prompt = (
        f"{hist_text}\nQuestion: {truncate_text(question, MAX_DOC_CHARS)}\n\n"
        f"Context documents:\n{sources_text}\n\n"
        "Answer with the shortest and most precise response possible."
    )

    if len(user_prompt) > MAX_PROMPT_CHARS:
        user_prompt = truncate_text(user_prompt, MAX_PROMPT_CHARS)

    return system, user_prompt

# ============ API ============
@app.route("/api/query", methods=["POST"])
def api_query():
    data = request.json or {}
    q = data.get("q", "").strip()
    session_id = data.get("session_id", "default")

    if not q:
        return jsonify({"error": "No query provided"}), 400

    q_lower = q.lower()
    if q_lower in {"stop", "exit", "okay stop", "ok stop", "wait"}:
        return jsonify({"answer": "[stopped]", "retrieved": [], "history": HISTORY.get(session_id, [])})

    if q_lower in {"clear", "clear history", "reset"}:
        HISTORY[session_id] = []
        gc.collect()
        return jsonify({"answer": "History cleared.", "retrieved": [], "history": []})

    # Step 0: Admission/Cutoff priority search
    admission_keywords = {"cutoff", "cut off" , "rank", "cet", "marks"}
    if any(word in q_lower for word in admission_keywords):
        cutoff_answer = search_cutoff_embeddings(q)  # Markdown string
        if cutoff_answer:
            hist = HISTORY.get(session_id, [])
            hist.append({"q": q, "a": cutoff_answer})
            HISTORY[session_id] = hist[-10:]
            return jsonify({"answer": cutoff_answer, "retrieved": [], "history": HISTORY[session_id]})
        # else fallback continues...

    # Step 1: Semantic JSON lookup
    json_answer = search_json_embeddings(q)
    if json_answer:
        hist = HISTORY.get(session_id, [])
        hist.append({"q": q, "a": json_answer})
        HISTORY[session_id] = hist[-10:]
        return jsonify({"answer": json_answer, "retrieved": [], "history": HISTORY[session_id]})

    # Step 2: General FAISS + Groq
    try:
        retrieved = retrieve(q, top_k=3)
    except Exception as e:
        logger.exception("Retrieval failed: %s", e)
        return jsonify({"error": f"Retrieval failed: {str(e)}"}), 500

    hist = HISTORY.get(session_id, [])
    system, user_prompt = build_prompt(q, retrieved, hist)

    try:
        logger.info("Calling Groq: %s", q[:80])
        answer = run_async(groq_generate_async(system, user_prompt, max_tokens=300, temperature=0.1))
    except Exception as e:
        logger.error("Groq API error: %s", e)
        return jsonify({"error": "Groq API error"}), 502

    hist.append({"q": q, "a": answer})
    HISTORY[session_id] = hist[-10:]

    return jsonify({"answer": answer, "retrieved": retrieved, "history": HISTORY[session_id]})

@app.route("/api/history", methods=["GET"])
def api_history():
    session_id = request.args.get("session_id", "default")
    return jsonify(HISTORY.get(session_id, []))

@app.route("/api/health", methods=["GET"])
def api_health():
    return jsonify({
        "status": "ok",
        "faiss_loaded": faiss_index is not None,
        "cutoff_loaded": cutoff_index is not None,
        "qa_count": len(qa_data)
    })

@app.route("/")
def frontend_index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/<path:path>")
def static_proxy(path):
    return send_from_directory(app.static_folder, path)

if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    print(f"Running Flask server on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)
