
````markdown
# AI Interview Engine (English) 🚀

This repository provides an AI-powered interview evaluation system that analyzes candidate answers based on a provided resume. It supports both **RAGPython** (v1) and **Anchor RAG** (v2) approaches.

---

## 📌 Overview

The engine allows:

* Automatic scoring of interview answers (0–5) based on resume alignment
* Detection of gibberish or low-information responses
* Mode-based evaluation:

  * **EASY** – lenient
  * **NORMAL** – balanced
  * **HARD** – strict, resume-only

It uses **RAG (Retrieval-Augmented Generation)** to retrieve relevant resume chunks and an LLM (Gemma 2B or phi3) for scoring and feedback.

---

## 🔹 Features

* Resume PDF parsing via `pdfplumber`
* Chunk-based vector indexing (FAISS) for efficient RAG
* Sentence embeddings via `sentence-transformers/all-MiniLM-L6-v2`
* Low RAM and CPU-friendly optimizations for 8GB environments
* Gibberish detection and low-information answer detection
* Detailed NDJSON logs for traceability

---

## 📊 Improvements: RAGPython → Anchor RAG

| Item                      | RAGPython (v1)              | Anchor RAG (v2)                                          | Improvement                                                  |
| ------------------------- | --------------------------- | -------------------------------------------------------- | ------------------------------------------------------------ |
| LLM model                 | phi3                        | gemma:2b                                                 | CPU-friendly, stable on low-RAM systems                      |
| Text chunking             | 600 chars / 150 overlap     | 200 words / 40 overlap                                   | Smaller chunks → faster RAG, less memory                     |
| Vector index              | FAISS L2                    | FAISS IP with normalized embeddings                      | Cosine similarity for stable scoring                         |
| Similarity check          | Only deviation threshold    | Deviation + off-topic + low-information                  | More accurate evaluation                                     |
| Low-information detection | None                        | Tokens / hint words / lexical variety                    | Automatic score capping for weak answers                     |
| Gibberish detection       | None                        | Simple rules on short/meaningless strings                | Prevent meaningless answers from advancing                   |
| Prompt design             | LLM receives resume excerpt | Mode-specific prompts (EASY/NORMAL/HARD) + resume chunks | Better grading control                                       |
| Logging                   | Basic JSON                  | NDJSON with detailed metrics                             | Full trace of evaluation, similarity, penalties              |
| CPU/RAM tuning            | None                        | `_configure_runtime_for_low_ram_cpu()`                   | Stable on 8GB machines                                       |
| User experience           | Input → LLM → Score         | Input → Gibberish/Low-info → RAG → LLM → Score           | Avoid scoring meaningless responses, configurable difficulty |

> **💡 Summary:** Anchor RAG improves stability, safety, and scoring precision by adding pre-checks, mode-based evaluation, and better memory management.

---

## ⚙️ Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/resume_speak.git
cd resume_speak
````

2. Install dependencies (Python 3.10+ recommended):

```bash
pip install -r requirements.txt
```

3. Place the sample resume PDF in the repository root:

* [Download Sample Resume PDF](https://github.com/yourusername/resume_speak/raw/main/resume.pdf)

> ⚠️ **Important:** This is a **fully generic sample resume**. Do **not** upload real personal data to the repository.

4. Run the engine:

```bash
python anchor_rag_interview.py
```

or for the original RAGPython:

```bash
python Interview_Resume.py
```

---

## 📝 Usage

1. Select evaluation mode:

```
1 = EASY   (lenient)
2 = NORMAL (default)
3 = HARD   (strict resume-only)
```

2. Answer interview questions prompted by the system.
3. Receive score, feedback, and optional guidance for improving answers.
4. Logs are saved as NDJSON in the repository root:

```
interview_log_YYYYMMDD_HHMMSS.ndjson
```

---

## 💾 Logging

Each record contains:

* `question_index` & `question`
* `answer`
* `retrieved_chunks` (top-K)
* `similarity`
* `gibberish_detected` / `low_information_detected`
* `low_information_score_cap` applied
* `llm_feedback`
* `final score`
* `timestamp`

This allows full traceability of scoring decisions.

---

## 📄 Sample Questions

1. Can you introduce yourself?
2. What motivates you to apply for this position?
3. What are your strengths?
4. What achievements are you most proud of?
5. Can you describe your PC skills?

> Responses are evaluated based on alignment with resume content.

---

## 🔧 Customization

* Adjust `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K` in `anchor_rag_interview.py` for different retrieval granularity.
* Switch LLM by changing `OLLAMA_MODEL`.
* Modify scoring thresholds and low-information parameters for domain-specific tuning.

---

## 📚 References

* [Sentence Transformers](https://www.sbert.net/)
* [FAISS](https://github.com/facebookresearch/faiss)
* [Ollama API](https://ollama.com/)

---

## ⚠️ Notes

* Anchor RAG is designed for **English resumes** and interview answers.
* For low-RAM (<8GB) environments, the system reduces parallel threads automatically.
* Always keep your resume PDF updated to reflect the correct candidate data.

---

## 📝 License

This project is licensed under the **MIT License**.
For details, see the [LICENSE](LICENSE) file in this repository.

---

```
