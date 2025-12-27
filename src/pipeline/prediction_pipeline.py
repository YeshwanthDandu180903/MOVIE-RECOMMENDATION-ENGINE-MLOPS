import os
import sys
import pickle
import numpy as np
import pandas as pd
from scipy import sparse
from difflib import SequenceMatcher

from src.exception import MyException
from src.logger import logging
from src.constants import (
    TFIDF_VECTORIZER_PATH,
    TFIDF_MATRIX_PATH,
    COSINE_SIMILARITY_PATH
)


# -----------------------------
# Helper functions
# -----------------------------
def normalize_text(text: str) -> str:
    return text.lower().strip()


def seq_ratio(a, b):
    return SequenceMatcher(None, a, b).ratio()


def jaccard(a_tokens, b_tokens):
    A, B = set(a_tokens), set(b_tokens)
    if not A or not B:
        return 0
    return len(A & B) / len(A | B)


class MovieRecommender:
    def __init__(self):
        """
        Load trained recommender artifacts
        """
        try:
            logging.info("Loading recommender artifacts")

            # Load dataset
            self.df = self._load_latest_dataframe()

            # Normalize titles
            self.df["title_norm"] = self.df["title"].apply(normalize_text)
            self.df["title_tokens"] = self.df["title_norm"].apply(lambda x: x.split())

            # Handle ratings
            self.df["rating"] = self.df["rating"].fillna(self.df["rating"].mean())

            # Load cosine similarity
            self.cosine_sim = np.load(COSINE_SIMILARITY_PATH)

            logging.info("Recommender artifacts loaded successfully")

        except Exception as e:
            raise MyException(e, sys)

    def _load_latest_dataframe(self) -> pd.DataFrame:
        """
        Load latest ingested movie CSV
        """
        for root, _, files in os.walk("src/artifacts"):
            for file in files:
                if file == "movies.csv":
                    return pd.read_csv(os.path.join(root, file))
        raise Exception("movies.csv not found in artifacts")

    # -----------------------------
    # Movie matching logic (ADVANCED)
    # -----------------------------
    def find_movie(self, query):
        q = normalize_text(query)
        q_tokens = q.split()

        # SHORT TITLES
        if len(q) <= 4:
            exact = self.df[self.df["title_norm"] == q]
            if not exact.empty:
                return exact.iloc[0]["title"]

            sub = self.df[self.df["title_norm"].str.contains(q)]
            if not sub.empty:
                return sub.sort_values("vote_count", ascending=False).iloc[0]["title"]

            pre = self.df[self.df["title_norm"].str.startswith(q)]
            if not pre.empty:
                return pre.sort_values("vote_count", ascending=False).iloc[0]["title"]

            best_title, best_score = None, 0
            for _, row in self.df.iterrows():
                score = seq_ratio(q, row["title_norm"])
                if score > best_score:
                    best_score = score
                    best_title = row["title"]

            return best_title

        # NORMAL TITLES
        exact = self.df[self.df["title_norm"] == q]
        if not exact.empty:
            return exact.iloc[0]["title"]

        sub = self.df[self.df["title_norm"].str.contains(q)]
        if not sub.empty:
            return sub.sort_values("vote_count", ascending=False).iloc[0]["title"]

        best_title, best_score = None, 0
        for _, row in self.df.iterrows():
            score = (
                0.7 * jaccard(q_tokens, row["title_tokens"]) +
                0.3 * seq_ratio(q, row["title_norm"])
            )
            if score > best_score:
                best_score = score
                best_title = row["title"]

        return best_title

    # -----------------------------
    # Recommendation
    # -----------------------------
    def recommend(self, movie_name: str, top_n: int = 10) -> pd.DataFrame:
        try:
            logging.info(f"Generating recommendations for: {movie_name}")

            title = self.find_movie(movie_name)
            if title is None:
                raise Exception("Movie not found")

            idx = self.df.index[self.df["title"] == title][0]

            sim_scores = list(enumerate(self.cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

            movie_indices = [i[0] for i in sim_scores]

            return self.df.iloc[movie_indices][
                ["title", "genres", "rating", "poster_url"]
            ]

        except Exception as e:
            raise MyException(e, sys)
