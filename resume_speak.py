import os
import json
import hashlib
import datetime
import requests
import pdfplumber
import numpy as np
import faiss
from typing import List, Tuple
from sentence_transformers import SentenceTransformer

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "phi3"
TIMEOUT = 300

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 64
CHUNK_CHARS = 600
CHUNK_OVERLAP = 150
TOP_K = 2

EMBED_CACHE_DIR = ".embed_cache"
INDEX_CACHE_FILE = os.path.join(EMBED_CACHE_DIR, "faiss_index.index")
QUESTION_EMB_FILE = os.path.join(EMBED_CACHE_DIR, "questions.npy")

EVAL_MODE = "NORMAL"

os.makedirs(EMBED_CACHE_DIR, exist_ok=True)

DEVICE = "cpu"
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)
print(f"Device: {DEVICE}")


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def load_pdf_text(path: str = "resume.pdf") -> str:
    if not os.path.exists(path):
        return ""
    text = ""
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            t = page.extract_text()
            if t:
                text += f"[page:{i}]\n{t}\n"
    return text


def split_text_char_based(text: str, chunk_size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> List[dict]:
    chunks = []
    n = len(text)
    i = 0
    idx = 0
    while i < n:
        end = min(i + chunk_size, n)
        chunk_text = text[i:end]
        chunks.append({
            "text": chunk_text,
            "meta": {
                "chunk_id": idx,
                "start_char": i,
                "end_char": end
            }
        })
        if end == n:
            break
        i += chunk_size - overlap
        idx += 1
    return chunks


def build_vector_index_cached(chunks: List[dict]) -> faiss.IndexFlatL2:
    if os.path.exists(INDEX_CACHE_FILE):
        try:
            return faiss.read_index(INDEX_CACHE_FILE)
        except Exception:
            pass

    texts = [c["text"] for c in chunks]
    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True
    ).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, INDEX_CACHE_FILE)
    return index


def load_or_build_question_embeddings(questions: List[str]) -> np.ndarray:
    if os.path.exists(QUESTION_EMB_FILE):
        try:
            return np.load(QUESTION_EMB_FILE)
        except Exception:
            pass

    q_emb = embedding_model.encode(
        questions,
        convert_to_numpy=True,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True
    ).astype("float32")

    np.save(QUESTION_EMB_FILE, q_emb)
    return q_emb


def search_chunks_precomputed(q_idx: int, chunks: List[dict], index: faiss.IndexFlatL2,
                              question_embeddings: np.ndarray, top_k: int = TOP_K) -> str:
    q_emb = question_embeddings[q_idx].reshape(1, -1)
    k = min(top_k, len(chunks))
    D, I = index.search(q_emb, k)

    lines = []
    for idx in I[0]:
        if 0 <= idx < len(chunks):
            c = chunks[idx]
            lines.append(f"[chunk:{c['meta']['chunk_id']}] {c['text'][:400]}")
    return "\n".join(lines)


def ask_ollama(prompt: str) -> str:
    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": MODEL, "prompt": prompt, "stream": False},
            timeout=TIMEOUT
        )
        r.raise_for_status()
        data = r.json()
        if "response" in data:
            return data["response"]
        return json.dumps(data, ensure_ascii=False)
    except Exception as e:
        return f"Ollama error: {e}"


def build_feedback_prompt(rag_text: str, question: str, user_answer: str, mode: str = "NORMAL") -> str:
    rule = (
        "Evaluate mainly based on the resume. Extra information is allowed but not scored. "
        "If the answer contradicts the resume, score=0.\n"
    )
    if mode == "HARD":
        rule = "Use ONLY the resume. Any information not found in the resume must result in score=0.\n"
    if mode == "EASY":
        rule = "Evaluate based on the resume, but allow reasonable extra information unless it contradicts the resume.\n"

    return (
        "You are an interviewer evaluation assistant.\n"
        "Evaluate the answer based on the resume excerpt.\n"
        "Give a numeric score from 0 to 5 and one short feedback sentence.\n"
        f"{rule}"
        f"Resume excerpt:\n{rag_text}\n\n"
        f"Question:\n{question}\n\n"
        f"Applicant answer:\n{user_answer}\n"
    )


def deviation_detector(answer: str, rag_text: str, threshold: float = 0.35) -> Tuple[bool, float]:
    try:
        a_emb = embedding_model.encode(
            [answer], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")

        r_emb = embedding_model.encode(
            [rag_text[:2000]], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")

        sim = float(np.dot(a_emb, r_emb.T))
        return sim < threshold, sim
    except Exception:
        return False, 0.0


class InterviewLogger:
    def __init__(self, pdf_text: str, out_dir: str = "."):
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = os.path.join(out_dir, f"interview_log_{now}.ndjson")
        self.file = open(self.filepath, "a", encoding="utf-8", buffering=1024 * 64)

        header = {
            "timestamp": now,
            "resume_hash": sha256_text(pdf_text)
        }
        self.file.write(json.dumps({"meta": header}, ensure_ascii=False) + "\n")

    def write(self, record: dict):
        self.file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def close(self):
        self.file.flush()
        self.file.close()


def run_interview():
    global EVAL_MODE

    print("===== AI Interview Engine (English) =====")
    print("Select evaluation mode:")
    print("1 = EASY   (lenient)")
    print("2 = NORMAL (default)")
    print("3 = HARD   (strict resume-only)")

    mode = input("Enter 1 / 2 / 3: ").strip()

    if mode == "1":
        EVAL_MODE = "EASY"
    elif mode == "3":
        EVAL_MODE = "HARD"
    else:
        EVAL_MODE = "NORMAL"

    print(f"Evaluation mode: {EVAL_MODE}")

    pdf_text = load_pdf_text()
    if not pdf_text:
        print("resume.pdf not found")
        return

    chunks = split_text_char_based(pdf_text)
    print("Chunks:", len(chunks))

    index = build_vector_index_cached(chunks)

    questions = [
        "Can you introduce yourself?",
        "What motivates you to apply for this position?",
        "What are your strengths?",
        "What achievements are you most proud of?",
        "Can you describe your PC skills?"
    ]

    question_embeddings = load_or_build_question_embeddings(questions)

    logger = InterviewLogger(pdf_text)

    need_correction = False

    for i, q in enumerate(questions):
        print("\nInterviewer:", q)
        user_answer = input("You: ").strip()

        rag_text = search_chunks_precomputed(i, chunks, index, question_embeddings)

        if need_correction:
            correction_text = (
                "Your previous answer deviated from the resume. "
                "For this question, answer strictly based on the resume content.\n\n"
                "Hint: mention your real experience, skills, or achievements from your resume. "
                "Keep it concise and specific.\n\n"
            )
        else:
            correction_text = ""

        feedback_prompt = correction_text + build_feedback_prompt(
            rag_text, q, user_answer, EVAL_MODE
        )

        ai_feedback_raw = ask_ollama(feedback_prompt)

        # Interpret AI text output without requiring strict JSON format
        # Expected plain text response like: "Score: 4, Feedback: ..."
        score = {"score": 0, "feedback": ai_feedback_raw.strip()}
        if "score" in ai_feedback_raw.lower():
            # optional parse
            import re
            m = re.search(r"score\s*[:=]\s*(\d+)", ai_feedback_raw, re.IGNORECASE)
            if m:
                score["score"] = int(m.group(1))

        is_dev, sim = deviation_detector(user_answer, rag_text)

        print("\nAI Feedback:", score.get("feedback"))
        print("Score:", score.get("score"))

        if is_dev:
            print(f"Note: Your answer seems a bit off from your resume details (similarity={sim:.2f}).")
            print("Let's try to keep the next answer aligned with your resume so you can score better.")
            need_correction = True
        else:
            need_correction = False

        record = {
            "question": q,
            "answer": user_answer,
            "rag": rag_text[:800],
            "ai_feedback": score,
            "deviation": {"flag": is_dev, "similarity": sim},
            "correction_next": need_correction
        }

        logger.write(record)

    logger.close()
    print("\nLog saved:", logger.filepath)


if __name__ == "__main__":
    run_interview()