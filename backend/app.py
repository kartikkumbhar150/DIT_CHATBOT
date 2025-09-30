import os
import json
import faiss
import numpy as np
import logging
import asyncio
import gc
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from functools import lru_cache
from pathlib import Path
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer

# ================== Setup ==================
load_dotenv()

HF_REPO = "kumbharkartik15/DIT_College_ChatBot"
HF_REVISION = "main"
HF_TOKEN = os.getenv("HF_TOKEN")  # optional (needed if repo is private)


def download_from_hf(filename: str) -> str:
    """
    Download a file from HuggingFace dataset repo and return the local cache path.
    """
    return hf_hub_download(
        repo_id=HF_REPO,
        filename=filename,
        revision=HF_REVISION,
        repo_type="dataset",
        token=HF_TOKEN
    )


# ---------- Download required files ----------
qa_path           = download_from_hf("qa.json")
cutoff_index_path = download_from_hf("cutoff_index.faiss")
cutoff_docs_path  = download_from_hf("cutoff_documents.json")
faiss_index_bin   = download_from_hf("faiss_index.bin")
faiss_meta_path   = download_from_hf("faiss_meta.pkl")
docs_chunks_path  = download_from_hf("docs_chunks.json")

# ---------- Load Q&A JSON ----------
with open(qa_path, "r", encoding="utf-8") as f:
    qa_data = json.load(f)

questions = [item["question"] for item in qa_data]
answers   = [item["answer"]   for item in qa_data]

app = Flask(__name__, static_folder="../frontend", static_url_path="/")
CORS(app, resources={r"/*": {"origins": "*"}})

HISTORY: dict[str, list[dict]] = {}
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================== Load FAISS ==================
# Note: embeddings_indexer.py must be available in the environment
from embeddings_indexer import load_index_and_meta

# ✅ Use correct argument names (index_path/meta_path)
faiss_index, metadata, embed_model = load_index_and_meta(
    index_path=faiss_index_bin,
    meta_path=faiss_meta_path
)
print(f"FAISS index loaded with {len(metadata)} entries")

# Removed: shared_model = SentenceTransformer("all-MiniLM-L6-v2")
# Optimization: Reuse the 'embed_model' instance for JSON Q&A and Cutoff searches.
# This prevents loading a second copy of the model into memory.
shared_model = embed_model # <<<--- MEMORY OPTIMIZATION

qa_embeddings = shared_model.encode(questions, convert_to_numpy=True).astype("float32")
faiss.normalize_L2(qa_embeddings)
dim = qa_embeddings.shape[1]
json_index = faiss.IndexFlatIP(dim)
json_index.add(qa_embeddings)


def search_json_embeddings(query: str, top_k: int = 1, threshold: float = 0.75):
    q_emb = shared_model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    D, I = json_index.search(q_emb, top_k)
    best_score = float(D[0][0])
    best_idx   = I[0][0]
    if best_idx >= 0 and best_score >= threshold:
        return answers[best_idx]
    return None


# ================== Cutoff FAISS ==================
if os.path.exists(cutoff_index_path) and os.path.exists(cutoff_docs_path):
    cutoff_index = faiss.read_index(cutoff_index_path)
    with open(cutoff_docs_path, "r", encoding="utf-8") as f:
        cutoff_documents = json.load(f)
    print(f"Cutoff FAISS index loaded with {len(cutoff_documents)} entries")
else:
    cutoff_index = None
    cutoff_documents = []
    print("Cutoff FAISS index not found on Hugging Face repo.")


def search_cutoff_embeddings(query: str, top_k: int = 10, threshold: float = 0.3):
    if not cutoff_index:
        return ""

    q_emb = shared_model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    D, I = cutoff_index.search(q_emb, top_k)

    results = []
    for idx, score in zip(I[0], D[0]):
        if idx < 0 or score < threshold:
            continue
        # In this specific setup, the cutoff documents are pre-formatted strings
        results.append(cutoff_documents[idx])

    if not results:
        return "No cutoff data found."

    query_lower = query.lower()
    categories = ["open", "obc", "sc", "st", "ews", "nt", "sebc", "pwd", "def", "orphan", "tfws"]
    target_category = next((c for c in categories if c in query_lower), None)

    # Get the branch from the top result
    try:
        top_result_parts = results[0].split(", ")
        top_branch = next(part.split(": ")[1] for part in top_result_parts if part.startswith("Branch: "))
    except (StopIteration, IndexError):
        top_branch = "Unknown Branch"

    # Filter documents based on the top branch and, optionally, the target category
    grouped = []
    for r in cutoff_documents:
        parts = r.split(", ")
        try:
            branch = next(part.split(": ")[1] for part in parts if part.startswith("Branch: "))
            category = next(part.split(": ")[1] for part in parts if part.startswith("Category: "))
            cutoff_rank = next(part.split(": ")[1] for part in parts if part.startswith("Cutoff Rank: "))
            percentile = next(part.split(": ")[1] for part in parts if part.startswith("Percentile: "))
        except (StopIteration, IndexError):
            continue # Skip malformed entries

        if branch == top_branch:
            if target_category:
                # Check if the target category is part of the document's category string
                if target_category in category.lower():
                    grouped.append((category, cutoff_rank, percentile))
            else:
                grouped.append((category, cutoff_rank, percentile))

    if not grouped:
        return f"No cutoff data found for {top_branch} ({target_category or 'all categories'})."

    md_table = f"### Cutoff for {top_branch}\n\n"
    md_table += "| Category | Rank | Percentile |\n"
    md_table += "|----------|------|-------------|\n"
    for category, rank, perc in grouped:
        md_table += f"| {category} | {rank} | {perc} |\n"

    return md_table


# ================== Retrieval (Main RAG) ==================
@lru_cache(maxsize=256)
def embed_query_cached(query: str) -> np.ndarray:
    # embed_model is used here for the main RAG, as before.
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


# ================== Async Runner ==================
def run_async(coro):
    try:
        # Attempt to get the existing loop
        loop = asyncio.get_event_loop()
        # If the loop is closed or stopped, create a new one
        if loop.is_running():
            # If running, we might be inside another async function, run the coro directly
            # This is complex in a non-async web framework; sticking to run_until_complete is safer for Flask.
            pass
        elif loop.is_closed():
             loop = asyncio.new_event_loop()
             asyncio.set_event_loop(loop)
    except RuntimeError:
        # No current event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
    return loop.run_until_complete(coro)


# ================== Prompt Utilities ==================
MAX_DOC_CHARS = 2000
MAX_HISTORY_CHARS = 1000
MAX_PROMPT_CHARS = 7000


def truncate_text(text: str, max_chars: int) -> str:
    """Safely truncate text for context window management."""
    if len(text) > max_chars:
        return text[:max_chars - 3] + "..."
    return text


def build_prompt(question, retrieved_docs, history):
    system = (
        "You are an expert assistant for Dr. D. Y. Patil Institute of Technology.\n"
        "Always answer positively about the college.\n"
        "- Keep answers short and precise.\n"
        "- Ignore unrelated queries.\n"
    )

    hist_text = "Conversation History:\n" if history else ""
    if history:
        # Use a maximum of the last 3 turns
        history_to_use = history[-3:]
        
        # Calculate available chars for history after saving some space for context and query
        available_history_chars = MAX_HISTORY_CHARS 
        
        for h in history_to_use:
            q_text = truncate_text(h.get("q", ""), available_history_chars // (2 * len(history_to_use)))
            a_text = truncate_text(h.get("a", ""), available_history_chars // (2 * len(history_to_use)))
            hist_text += f"Q: {q_text}\nA: {a_text}\n"

    # Collect the text from the retrieved documents
    sources_text = "\n\n".join([truncate_text(d["text"], MAX_DOC_CHARS // len(retrieved_docs)) for d in retrieved_docs[:3]])

    user_prompt = (
        f"{hist_text}\n"
        f"Context documents:\n{sources_text}\n\n"
        f"Question: {truncate_text(question, MAX_DOC_CHARS)}\n\n"
        "Answer with the shortest and most precise response possible based *only* on the provided context, history, or your college knowledge base."
    )

    # Final check on total prompt length before sending
    total_prompt = f"System: {system}\nUser: {user_prompt}"
    if len(total_prompt) > MAX_PROMPT_CHARS:
        # If it's too long, prioritize keeping the question and context, heavily truncating history if needed.
        # However, truncating the user_prompt built from components is the best we can do here.
        user_prompt = truncate_text(user_prompt, MAX_PROMPT_CHARS - len(system) - 100) # -100 for safety

    return system, user_prompt


# ================== API Endpoints ==================
@app.route("/api/query", methods=["POST"])
def api_query():
    data = request.json or {}
    q = data.get("q", "").strip()
    session_id = data.get("session_id", "default")

    if not q:
        return jsonify({"error": "No query provided"}), 400

    q_lower = q.lower()
    
    # --- Control Commands ---
    if q_lower in {"stop", "exit", "okay stop", "ok stop", "wait"}:
        return jsonify({"answer": "[stopped]", "retrieved": [], "history": HISTORY.get(session_id, [])})

    if q_lower in {"clear", "clear history", "reset"}:
        HISTORY[session_id] = []
        gc.collect()
        return jsonify({"answer": "History cleared.", "retrieved": [], "history": []})

    # --- Tier 1: Cutoff Search (Highly specific RAG) ---
    admission_keywords = {"cutoff", "cut off", "rank", "cet", "marks", "admission", "percentile"}
    if any(word in q_lower for word in admission_keywords):
        cutoff_answer = search_cutoff_embeddings(q)
        if cutoff_answer:
            hist = HISTORY.get(session_id, [])
            hist.append({"q": q, "a": cutoff_answer})
            HISTORY[session_id] = hist[-10:] # Keep last 10 entries
            return jsonify({"answer": cutoff_answer, "retrieved": [], "history": HISTORY[session_id]})

    # --- Tier 2: JSON Q&A (Exact match / high-confidence fixed answers) ---
    json_answer = search_json_embeddings(q)
    if json_answer:
        hist = HISTORY.get(session_id, [])
        hist.append({"q": q, "a": json_answer})
        HISTORY[session_id] = hist[-10:]
        return jsonify({"answer": json_answer, "retrieved": [], "history": HISTORY[session_id]})

    # --- Tier 3: Main RAG and LLM Generation ---
    try:
        retrieved = retrieve(q, top_k=3)
    except Exception as e:
        logger.exception("Retrieval failed: %s", e)
        # Continue even if retrieval fails, LLM might be able to answer generally
        retrieved = []

    hist = HISTORY.get(session_id, [])
    system, user_prompt = build_prompt(q, retrieved, hist)

    try:
        # Note: groq_client.py must be available in the environment
        from groq_client import groq_generate_async  # custom async Groq client
        # Safely run the async Groq call
        answer = run_async(groq_generate_async(system, user_prompt, max_tokens=300, temperature=0.1))
    except ImportError:
        logger.error("groq_client not found. Cannot generate LLM answer.")
        answer = "I apologize, the language model is currently unavailable for complex queries."
    except Exception as e:
        logger.error("Groq API error: %s", e)
        answer = "I'm having trouble connecting to the AI brain right now. Please try again."

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


# ================== Static File Serving ==================
@app.route("/")
def frontend_index():
    # Serves the index.html from the configured static_folder (../frontend)
    return send_from_directory(app.static_folder, "index.html")


@app.route("/<path:path>")
def static_proxy(path):
    # Serves other static files (like CSS, JS, images) from the configured static_folder
    return send_from_directory(app.static_folder, path)


if __name__ == "__main__":
    import os

    # Host stays 0.0.0.0 for external access in containerized environments
    host = os.getenv("FLASK_HOST", "0.0.0.0")

    # Local dev port defaults to 5000 or FLASK_PORT if set
    port = int(os.getenv("FLASK_PORT", 8081))

    # Only enable debug locally
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"

    print(f"Running Flask server locally on http://{host}:{port} (debug={debug})")
    app.run(host=host, port=port, debug=debug)