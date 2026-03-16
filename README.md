# AI Interview Engine (Local LLM + RAG)

A resume-based AI interview simulator with hallucination detection using local LLMs.

This project evaluates interview answers based on a resume using Retrieval-Augmented Generation (RAG) and detects deviations from the resume using embedding similarity.

---

## Why this project

Large Language Models sometimes evaluate answers even when they contradict a candidate's resume.

This project explores a simple reliability layer for AI evaluation:

* Retrieve relevant resume context using RAG
* Evaluate answers with a local LLM
* Detect when answers deviate from the resume

The system runs entirely with local models.

---

## Architecture

```
resume.pdf
     │
     ▼
Text Chunking
     │
     ▼
Embedding (SentenceTransformers)
     │
     ▼
FAISS Vector Search
     │
     ▼
Relevant Resume Context
     │
     ▼
LLM Evaluation (via Ollama)
     │
     ▼
Score + Feedback
     │
     ▼
Deviation Detector (embedding similarity)
     │
     ▼
Interview Log (NDJSON)
```

---

## Features

* Resume-based RAG retrieval
* Local LLM evaluation
* Answer scoring (0–5)
* Deviation detection from resume content
* Cached FAISS vector index
* Precomputed question embeddings
* Structured NDJSON interview logs

---

## Tech Stack

* Python
* FAISS
* SentenceTransformers
* Ollama
* Local LLMs

Supported models include:

* Phi-3
* Mistral
* Llama3

---

## How it works

1. The system loads `resume.pdf`.
2. The resume is split into overlapping text chunks.
3. Each chunk is converted into embeddings.
4. FAISS builds a vector search index.
5. For each interview question:

   * Relevant resume chunks are retrieved
   * The answer is evaluated by the LLM
   * A score and feedback are generated
6. A similarity check detects if the answer deviates from the resume.
7. All interactions are logged.

---

## Evaluation Modes

Three evaluation modes are available:

* EASY – allows reasonable extra information
* NORMAL – evaluates mainly based on the resume
* HARD – strictly resume-only evaluation

---

## Run the project

Install dependencies:

```
pip install -r requirements.txt
```

Run a local model with Ollama:

```
ollama run phi3
```

Start the interview engine:

```
python interview_engine.py
```

---

## Output

Each interview session generates a structured log file:

```
interview_log_YYYYMMDD_HHMMSS.ndjson
```

Each entry contains:

* question
* user answer
* retrieved resume context
* AI score
* feedback
* deviation detection

---

## Project Goal

This project experiments with a simple AI reliability layer:

LLM generation + retrieval grounding + deviation detection.

Instead of relying purely on the model, the system checks whether answers remain consistent with source data.

---

## License

MIT License
