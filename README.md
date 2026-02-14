# Hybrid Job Recommender System  
### NLP + Transformer-Based Semantic Search

---

## Overview

This project implements a **hybrid content-based job recommender** that ranks job postings based on a user’s skills or role description.

It combines:

- **TF-IDF with n-grams** for lexical relevance  
- **Sentence-BERT (SBERT)** for semantic similarity  

Results are ranked using a weighted hybrid similarity score.

---

## Architecture

React Frontend

|

Flask API (/recommend)

|

Hybrid Recommender (TF-IDF + SBERT)

|

Precomputed Artifacts


Training is performed offline.  
Inference loads serialized artifacts at backend startup.

---

## Methodology

### TF-IDF (Sparse Representation)

- `ngram_range = (1, 3)`
- `max_features = 30000`
- `min_df = 5`
- Cosine similarity in high-dimensional sparse space

Captures exact skill and phrase-level matches (e.g., "machine learning engineer").

---

### SBERT (Dense Representation)

- Model: `all-MiniLM-L6-v2`
- 384-dimensional sentence embeddings
- Cosine similarity after normalization

Captures contextual and semantic similarity.

---

### Hybrid Scoring

Score = 0.6 × TF-IDF + 0.4 × SBERT

Balances precise keyword matching with semantic flexibility.

---

## Tech Stack

**Backend**
- Flask
- Scikit-learn
- Sentence-Transformers
- NumPy / Pandas

**Frontend**
- React

---

## Computational Complexity

Let:
- `N` = number of jobs  
- `D` = embedding dimension  

Per query:
- TF-IDF → `O(N)`
- SBERT → `O(N × D)`

Scalable via vector indexing (e.g., FAISS).

---

## Key Highlights

- Hybrid lexical + semantic ranking  
- REST API–based inference service  
- Clear separation of training and inference  
- Full-stack integration (React + Flask)  
- Honours-level NLP and transformer application  

---

## Conclusion

This system integrates sparse TF-IDF modeling with dense transformer embeddings to build a robust, production-ready hybrid recommender aligned with advanced AIML coursework.
