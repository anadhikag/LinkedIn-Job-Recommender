import os
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class JobRecommender:
    """
    Handles artifact loading and hybrid similarity inference.
    Loads models once at initialization to ensure production readiness.
    """

    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = artifacts_dir
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.job_embeddings = None
        self.df_jobs = None
        self.sbert_model = None

        self._load_artifacts()

    def _load_artifacts(self):
        try:
            logger.info("Loading TF-IDF artifacts...")
            self.tfidf_vectorizer = joblib.load(
                os.path.join(self.artifacts_dir, "tfidf_vectorizer.pkl")
            )
            self.tfidf_matrix = joblib.load(
                os.path.join(self.artifacts_dir, "tfidf_matrix.pkl")
            )

            logger.info("Loading job embeddings and metadata...")
            self.job_embeddings = np.load(
                os.path.join(self.artifacts_dir, "job_embeddings.npy")
            )
            self.df_jobs = pd.read_csv(
                os.path.join(self.artifacts_dir, "jobs.csv")
            )

            # Normalize embeddings once at load time for faster cosine similarity
            # This allows us to use dot product or simpler similarity measures if needed,
            # and ensures consistency with normalized query vectors.
            self.job_embeddings = normalize(self.job_embeddings)

            logger.info("Loading SBERT model...")
            self.sbert_model = SentenceTransformer(
                "all-MiniLM-L6-v2"
            )

            logger.info("All recommender artifacts loaded successfully.")

        except Exception as e:
            logger.exception("Error loading artifacts.")
            raise e

    def recommend(self, query_text: str, top_k: int = 10) -> list:
        """
        Computes hybrid similarity:
        final_score = 0.6 * TF-IDF + 0.4 * SBERT
        """

        if not isinstance(query_text, str) or not query_text.strip():
            return []

        # Safety cap based on dataset size
        top_k = min(max(top_k, 1), len(self.df_jobs))

        # 1️⃣ TF-IDF similarity
        query_tfidf = self.tfidf_vectorizer.transform([query_text])
        tfidf_sim = cosine_similarity(query_tfidf, self.tfidf_matrix).flatten()

        # 2️⃣ SBERT similarity
        query_embedding = self.sbert_model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        sbert_sim = cosine_similarity(
            query_embedding, self.job_embeddings
        ).flatten()

        # 3️⃣ Hybrid scoring (weighted average)
        final_scores = (0.6 * tfidf_sim) + (0.4 * sbert_sim)

        # 4️⃣ Ranking
        top_indices = np.argsort(final_scores)[::-1][:top_k]

        recommendations = []
        for idx in top_indices:
            job_row = self.df_jobs.iloc[idx]

            recommendations.append(
                {
                    "job_id": str(job_row.get("job_id", "")),
                    "title": str(job_row.get("title", "N/A")),
                    "company": str(job_row.get("company_name", "N/A")),
                    "location": str(job_row.get("location", "N/A")),
                    "skills": str(job_row.get("skills_desc", "")),
                    # Convert to percentage (0-100) for UI friendliness
                    "score": round(float(final_scores[idx]) * 100, 2),
                }
            )

        return recommendations
    