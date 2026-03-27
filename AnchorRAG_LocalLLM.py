import os

def _configure_runtime_for_low_ram_cpu() -> int:
    try:
        cpu = os.cpu_count() or 4
    except NotImplementedError:
        cpu = 4
    threads = max(1, min(6, max(2, cpu // 2)))
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(key, str(threads))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return threads


_CPU_THREADS = _configure_runtime_for_low_ram_cpu()

import re
import json
import time
import uuid
import datetime
import textwrap
import requests
import pdfplumber
import faiss
import numpy as np

from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer

RESUME_PDF = "resume.pdf"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma:2b"

CHUNK_SIZE = 200
CHUNK_OVERLAP = 40
TOP_K = 2

EMBED_BATCH_SIZE = 8

OLLAMA_NUM_PREDICT_EVAL = 150
OLLAMA_NUM_PREDICT_QUESTION = 96

OFF_TOPIC_THRESHOLD_BY_MODE = {
    "EASY": 0.18,
    "NORMAL": 0.24,
    "HARD": 0.30,
}
OFF_TOPIC_SCORE_CAP_BY_MODE = {
    "EASY": 2,
    "NORMAL": 1,
    "HARD": 0,
}

GIBBERISH_MIN_TOKENS = 2
INTERVIEW_HINT_WORDS = {
    "i", "my", "me", "we", "our", "experience", "experienced", "background", "worked",
    "work", "career", "project", "projects", "skill", "skills", "strength", "strengths",
    "achievement", "achievements", "motivate", "motivated", "responsible", "team",
    "managed", "developed", "built", "learned", "years",
}
COMMON_VERBS = {
    "am", "is", "are", "was", "were", "be", "have", "has", "had", "do", "did", "can",
    "will", "work", "worked", "manage", "managed", "build", "built", "develop", "developed",
    "lead", "led", "support", "supported", "learn", "learned", "improve", "improved",
}
LOW_INFORMATION_SCORE_CAP = 2
LOW_INFO_MAX_TOKENS_WEAK = 5
LOW_INFO_UNIQUE_RATIO_WEAK = 0.72
LOW_INFO_MAX_HINT_HITS_WEAK = 1
LOW_INFO_TRIGGER_WEAK_SIGNALS = 2

LOG_DIR = "."
ENGINE_NAME = "AI Interview Engine (English)"

_embed_model = None


def get_embed_model():
    global _embed_model
    if _embed_model is None:
        import torch

        torch.set_num_threads(_CPU_THREADS)
        torch.set_num_interop_threads(1)
        print("Loading embedding model on device=cpu ...")
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")
        print("Embedding model loaded.")
    return _embed_model


def embed(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.array([])
    model = get_embed_model()
    return np.asarray(
        model.encode(
            texts,
            batch_size=EMBED_BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ),
        dtype=np.float32,
    )


def load_pdf_text(path: str = RESUME_PDF) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Resume PDF not found: {path}")
    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return "\n".join(pages)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    tokens = text.split()
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk = " ".join(tokens[start:end])
        chunks.append(chunk)
        if end >= len(tokens):
            break
        start = end - overlap
    return chunks


def build_faiss_index(chunks: List[str]) -> Tuple[faiss.IndexFlatIP, np.ndarray]:
    embeddings = embed(chunks)
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index, embeddings


def search_similar_chunks(
    question: str,
    answer: str,
    chunks: List[str],
    index: faiss.IndexFlatIP,
    embeddings: np.ndarray,
    top_k: int = TOP_K,
) -> List[Tuple[str, float]]:
    query_text = f"{question} {answer}"
    q_emb = embed([query_text])
    faiss.normalize_L2(q_emb)
    scores, idxs = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1:
            continue
        results.append((chunks[idx], float(score)))
    return results


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a_norm, b_norm))


def compute_answer_similarity(
    question: str,
    answer: str,
    resume_chunks: List[str],
    index: faiss.IndexFlatIP,
    embeddings: np.ndarray,
) -> float:
    hits = search_similar_chunks(question, answer, resume_chunks, index, embeddings, top_k=TOP_K)
    if not hits:
        return 0.0
    best_chunk, _ = hits[0]
    vecs = embed([answer, best_chunk])
    return cosine_similarity(vecs[0], vecs[1])


def _tokenize_alpha_words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text.lower())


def is_gibberish_answer(answer: str) -> Tuple[bool, str]:
    text = answer.strip()
    if not text:
        return True, "empty answer"

    tokens = _tokenize_alpha_words(text)
    if len(tokens) < GIBBERISH_MIN_TOKENS and len(text) <= 12:
        return True, "too short/incomplete"

    alpha_chars = sum(1 for ch in text if ch.isalpha())
    alpha_ratio = alpha_chars / max(1, len(text))
    if alpha_ratio < 0.55:
        return True, "non-linguistic string"

    compact = re.sub(r"\s+", "", text.lower())
    if re.search(r"(.)\1{4,}", compact):
        return True, "repetitive pattern"

    long_tokens = [t for t in tokens if len(t) >= 5]
    if long_tokens and all(not re.search(r"[aeiou]", t) for t in long_tokens):
        return True, "word-like noise"

    has_hint = any(t in INTERVIEW_HINT_WORDS for t in tokens)
    has_verb = any(t in COMMON_VERBS for t in tokens)
    short_token_ratio = sum(1 for t in tokens if len(t) <= 2) / max(1, len(tokens))
    weak_signals = 0
    if len(tokens) >= 3 and not has_verb:
        weak_signals += 1
    if len(tokens) >= 3 and short_token_ratio >= 0.34:
        weak_signals += 1
    if len(tokens) >= 3 and not has_hint:
        weak_signals += 1
    if weak_signals >= 2:
        return True, "semantically weak fragment"

    return False, ""


def is_low_information_answer(answer: str) -> Tuple[bool, str]:
    text = answer.strip()
    tokens = _tokenize_alpha_words(text)
    if len(tokens) < 3:
        return False, ""

    unique_ratio = len(set(tokens)) / max(1, len(tokens))
    hint_hits = sum(1 for t in tokens if t in INTERVIEW_HINT_WORDS)

    weak_signals = 0
    reasons = []

    if len(tokens) <= LOW_INFO_MAX_TOKENS_WEAK:
        weak_signals += 1
        reasons.append("very short answer")

    if len(tokens) >= 4 and unique_ratio <= LOW_INFO_UNIQUE_RATIO_WEAK:
        weak_signals += 1
        reasons.append("low lexical variety")

    if hint_hits <= LOW_INFO_MAX_HINT_HITS_WEAK:
        weak_signals += 1
        reasons.append("too little interview-relevant detail")

    if re.fullmatch(r"(i\s+am\s+\w+\.?)|(it\s+is\s+\w+\.?)", text.lower()):
        weak_signals += 1
        reasons.append("generic template-like response")

    if weak_signals >= LOW_INFO_TRIGGER_WEAK_SIGNALS:
        return True, ", ".join(reasons[:2])
    return False, ""


def call_ollama(
    prompt: str,
    *,
    model: str = OLLAMA_MODEL,
    max_tokens: int = OLLAMA_NUM_PREDICT_EVAL,
) -> str:
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                },
            },
            timeout=120,
        )
        resp.raise_for_status()
        result = resp.json()["response"]
    except Exception as e:
        print("Ollama error:", e)
        result = "Score: 0\nComment: Error"
    return result


def evaluate_answer(question: str, answer: str) -> str:
    prompt = f"""
You are an interview evaluator.

Question:
{question}

Answer:
{answer}

Evaluate the answer.

Return format:
Score: 0-5
Comment: short feedback
"""
    return call_ollama(prompt)


def build_eval_prompt(
    mode: str,
    question: str,
    answer: str,
    retrieved_chunks: List[Tuple[str, float]],
) -> str:
    mode_desc = {
        "EASY": "Be lenient. Give partial credit if the answer is generally reasonable or loosely related to the resume.",
        "NORMAL": "Evaluate fairly. Use the resume context but allow some reasonable extrapolation.",
        "HARD": "Be strict. Only give high scores if the answer closely matches the resume content.",
    }[mode]

    context_text = "\n\n".join(
        [f"[Chunk {i+1} | score={score:.3f}]\n{chunk}" for i, (chunk, score) in enumerate(retrieved_chunks)]
    )

    prompt = f"""
You are an AI interview evaluator.

Evaluation mode: {mode}
Guideline: {mode_desc}

You are given:
1) A resume excerpt (multiple chunks).
2) An interview question.
3) The candidate's answer.

Your task:
- Score the answer from 0 to 5.
- Base your judgment primarily on how well the answer aligns with the resume excerpt.
- In EASY mode, be generous and give partial credit if the answer is generally reasonable.
- In HARD mode, be strict and only give high scores if the answer clearly reflects the resume content.
- Do not infer missing meaning. Do not "repair" broken answers.
- If the answer is meaningless, random words, or incomplete so that intent is unclear, Score must be 0.
- Provide a short explanation.
- At the end, output a line starting with "Score:" followed by the numeric score only.

Resume excerpt:
{context_text}

Question:
{question}

Answer:
{answer}

Now evaluate the answer.
"""
    return textwrap.dedent(prompt).strip()


def parse_score(text: str) -> int:
    m = re.search(r"Score\s*:\s*(\d+)", text)
    if m:
        return int(m.group(1))
    return 0


def parse_score_from_response(text: str) -> Tuple[int, str]:
    return parse_score(text), text


def build_recenter_prompt(mode: str, question: str) -> str:
    templates = {
        "EASY": (
            "Your answer doesn’t seem to match the question very well, but let’s try again.\n"
            "Please tell me about your actual experience."
        ),
        "NORMAL": (
            "Your answer appears unrelated to the question.\n"
            "Please provide a response based on your real background and experience."
        ),
        "HARD": (
            "This answer does not reflect your resume or the interview question.\n"
            "Please respond strictly based on your documented experience."
        ),
    }
    return templates.get(mode, templates["NORMAL"])


def interview() -> None:
    for i in range(3):
        question_prompt = "Create a job interview question."
        question = call_ollama(question_prompt, max_tokens=OLLAMA_NUM_PREDICT_QUESTION)

        print("\nQuestion:", question)
        answer = input("Your answer: ")

        result = evaluate_answer(question, answer)
        score = parse_score(result)

        print(result)
        print("Score:", score)


QUESTIONS = [
    "Can you introduce yourself?",
    "What motivates you to apply for this position?",
    "What are your strengths?",
    "What achievements are you most proud of?",
    "Can you describe your PC skills?",
]


def make_log_path() -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(LOG_DIR, f"interview_log_{ts}.ndjson")


def log_event(f, data: Dict):
    f.write(json.dumps(data, ensure_ascii=False) + "\n")
    f.flush()


def select_mode() -> str:
    print(f"===== {ENGINE_NAME} =====")
    print("Select evaluation mode:")
    print("1 = EASY   (lenient)")
    print("2 = NORMAL (default)")
    print("3 = HARD   (strict resume-only)")
    mode_raw = input("Enter 1 / 2 / 3: ").strip()
    if mode_raw == "1":
        mode = "EASY"
    elif mode_raw == "3":
        mode = "HARD"
    else:
        mode = "NORMAL"
    print(f"Evaluation mode: {mode}")
    return mode


def main():
    mode = select_mode()

    resume_text = load_pdf_text(RESUME_PDF)
    chunks = chunk_text(resume_text, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"Chunks: {len(chunks)}")
    index, emb = build_faiss_index(chunks)

    log_path = make_log_path()
    with open(log_path, "w", encoding="utf-8") as lf:
        session_id = str(uuid.uuid4())

        for q_idx, question in enumerate(QUESTIONS, start=1):
            print()
            print(f"Interviewer: {question}")
            user_answer = input("You: ").strip()

            if not user_answer:
                print("No answer entered. Skipping evaluation for this question.")
                log_event(lf, {
                    "session_id": session_id,
                    "question_index": q_idx,
                    "question": question,
                    "answer": user_answer,
                    "mode": mode,
                    "note": "empty_answer",
                })
                continue

            gibberish_detected, gibberish_reason = is_gibberish_answer(user_answer)
            if gibberish_detected:
                forced_feedback = (
                    "Comment: Your answer appears meaningless or incomplete. "
                    "Please answer with clear and relevant content.\nScore: 0"
                )
                recenter_prompt = build_recenter_prompt(mode, question)
                print()
                print("AI Feedback:", forced_feedback)
                print("Score: 0")
                print(f"Prompt to return to topic: {recenter_prompt}")
                log_event(lf, {
                    "session_id": session_id,
                    "question_index": q_idx,
                    "question": question,
                    "answer": user_answer,
                    "mode": mode,
                    "gibberish_detected": True,
                    "gibberish_reason": gibberish_reason,
                    "low_information_detected": False,
                    "low_information_reason": "",
                    "low_information_score_cap": None,
                    "retrieved_chunks": [],
                    "similarity": 0.0,
                    "off_topic_threshold": OFF_TOPIC_THRESHOLD_BY_MODE.get(mode, 0.24),
                    "off_topic_penalty_applied": True,
                    "recenter_prompt": recenter_prompt,
                    "llm_feedback": forced_feedback,
                    "score": 0,
                    "timestamp": time.time(),
                })
                continue

            low_information_detected, low_information_reason = is_low_information_answer(user_answer)
            low_information_score_cap = LOW_INFORMATION_SCORE_CAP if low_information_detected else None

            similarity = compute_answer_similarity(question, user_answer, chunks, index, emb)
            off_topic_threshold = OFF_TOPIC_THRESHOLD_BY_MODE.get(mode, 0.24)
            recenter_prompt = None
            if similarity < off_topic_threshold:
                recenter_prompt = build_recenter_prompt(mode, question)

            retrieved = search_similar_chunks(question, user_answer, chunks, index, emb, top_k=TOP_K)
            prompt = build_eval_prompt(mode, question, user_answer, retrieved)
            llm_raw = call_ollama(prompt)
            score, full_feedback = parse_score_from_response(llm_raw)

            final_score = score
            low_information_penalty_applied = False
            if low_information_score_cap is not None:
                final_score = min(final_score, low_information_score_cap)
                low_information_penalty_applied = True

            off_topic_penalty_applied = False
            if similarity < off_topic_threshold:
                score_cap = OFF_TOPIC_SCORE_CAP_BY_MODE.get(mode, 1)
                final_score = min(final_score, score_cap)
                off_topic_penalty_applied = True

            print()
            print("AI Feedback:", full_feedback)
            print(f"Score: {final_score}")
            print(f"Note: Your answer seems a bit inconsistent with your resume details (similarity={similarity:.2f})."
                  if similarity < 0.4 else
                  f"Note: Your answer is reasonably aligned with your resume (similarity={similarity:.2f}).")
            if low_information_detected:
                print(f"Low-information note: {low_information_reason}")
            if recenter_prompt:
                print(f"Prompt to return to topic: {recenter_prompt}")
            if low_information_penalty_applied:
                print(f"Score cap applied due to low-information answer: {final_score}")
            if off_topic_penalty_applied:
                print(f"Score cap applied due to off-topic answer: {final_score}")

            log_event(lf, {
                "session_id": session_id,
                "question_index": q_idx,
                "question": question,
                "answer": user_answer,
                "mode": mode,
                "retrieved_chunks": [
                    {"text": c, "score": s} for c, s in retrieved
                ],
                "similarity": similarity,
                "off_topic_threshold": off_topic_threshold,
                "low_information_detected": low_information_detected,
                "low_information_reason": low_information_reason,
                "low_information_score_cap": low_information_score_cap,
                "low_information_penalty_applied": low_information_penalty_applied,
                "off_topic_penalty_applied": off_topic_penalty_applied,
                "recenter_prompt": recenter_prompt,
                "llm_feedback": full_feedback,
                "score": final_score,
                "timestamp": time.time(),
            })

    print(f"\nLog saved: {log_path}")


if __name__ == "__main__":
    main()