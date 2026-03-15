# resume-speaks

Let your resume answer interview questions with a local AI.

A local AI system that generates interview answers based on a resume using a Retrieval-Augmented Generation (RAG) pipeline and detects hallucinated responses by comparing generated answers with the original resume context.

---

## Concept

Large language models often produce confident answers that are not grounded in real data.  
In an interview setting, this can easily lead to misleading or inaccurate responses.

This project explores a simple idea:

**Ground AI answers strictly in a resume.**

The system retrieves relevant sections from a resume and generates answers using a local language model.  
It then checks whether the generated response deviates from the resume content.

If the answer drifts away from the resume context, the system flags it as a potential hallucination.

The goal is simple:

**Interview answers should remain consistent with the resume.**

---

## Architecture

The pipeline is intentionally simple and transparent.


Resume (text)
│
▼
Embedding
│
▼
Vector Search (retrieve relevant context)
│
▼
Local LLM generation
│
▼
Answer embedding
│
▼
Similarity check with retrieved context
│
▼
Deviation / hallucination detection


Key idea:

The generated answer is embedded and compared with the retrieved resume context using cosine similarity.  
If similarity falls below a threshold, the system marks the response as potentially hallucinated.

---

## Features

- Local LLM inference  
- Resume-based RAG  
- Interview question answering  
- Hallucination / deviation detection  
- Simple and transparent architecture  

---

## Why Local AI?

Running everything locally provides several advantages:

- Privacy (resumes contain personal information)  
- No API costs  
- Full control over the pipeline  
- Reproducible experiments  

---

## Example

**Input question**


What experience do you have with Python?


**Retrieved resume context**


Developed a local AI system using Python, embeddings, and vector search.


**Generated answer**


I built a local AI system in Python that retrieves information from my resume and generates grounded responses for interview questions.


The system then checks whether the generated answer remains consistent with the retrieved resume content.

---

## Possible Extensions

The concept can be applied beyond resumes:

- Legal document Q&A  
- Medical record assistants  
- Personal knowledge systems  
- Verified AI agents  

Any scenario where generated answers must remain grounded in source data.

---

## Motivation

This project started with a simple question:

**Can a resume answer interview questions by itself?**

While the implementation is small, the idea touches a broader challenge in AI:

How can we keep language models aligned with real data?

---

## 日本語説明

resume-speaks は、履歴書をもとに面接の質問へ回答するローカルAIシステムの実験的プロジェクトです。

大規模言語モデルは、ときにもっともらしい回答を生成しますが、その内容が必ずしも元データに基づいているとは限りません（いわゆるハルシネーション）。

このプロジェクトでは次のような仕組みを試しています。

1. 履歴書テキストを読み込む  
2. ベクトル検索により関連部分を取得する  
3. ローカルLLMで回答を生成する  
4. 生成された回答と履歴書内容の類似度を計算する  
5. 履歴書から逸脱している回答を検出する  

つまり、

**「履歴書に基づいた回答を生成し、その妥当性を確認する」**

というシンプルなアイデアです。

この仕組みは面接回答だけでなく、次のような用途にも応用できます。

- 法律文書の質問応答  
- 医療記録のサポートAI  
- 個人ナレッジベース  
- データに基づいたAIエージェント  

小さなプロジェクトですが、生成AIを元データに結びつける方法を探る試みです。

---

## License

MIT
