print(">>> Starting Movie Recommendation Backend...")

import os
import sys
import numpy as np
import pandas as pd
import unicodedata
from difflib import SequenceMatcher

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from src.exception import MyException
from src.logger import logging
from src.entity.estimator import MovieRecommenderEstimator
from src.cloud_storage.aws_storage import SimpleStorageService
from src.constants import (
    MODEL_BUCKET_NAME,
    MODEL_PUSHER_S3_KEY,
    TFIDF_VECTORIZER_FILE_NAME,
    TFIDF_MATRIX_FILE_NAME,
    COSINE_SIMILARITY_FILE_NAME,
    TFIDF_VECTORIZER_PATH,
    TFIDF_MATRIX_PATH,
    COSINE_SIMILARITY_PATH,
)


# =====================================================
# OPTIONAL: DOWNLOAD ARTIFACTS FROM S3 AT STARTUP
# =====================================================
def ensure_model_artifacts_from_s3(force_download: bool = True):
    """
    Always fetch artifacts from S3 when force_download is True
    to guarantee we serve the S3 version (no local fallback).
    """
    try:
        s3 = SimpleStorageService()
        downloads = [
            (TFIDF_VECTORIZER_FILE_NAME, TFIDF_VECTORIZER_PATH),
            (TFIDF_MATRIX_FILE_NAME, TFIDF_MATRIX_PATH),
            (COSINE_SIMILARITY_FILE_NAME, COSINE_SIMILARITY_PATH),
        ]

        for fname, local_path in downloads:
            local_path = str(local_path)

            if not force_download and os.path.exists(local_path):
                logging.info(f"Model artifact present locally, skipping download: {local_path}")
                continue

            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            s3_key = f"{MODEL_PUSHER_S3_KEY}/{fname}"
            logging.info(
                f"Downloading {s3_key} from s3://{MODEL_BUCKET_NAME} to {local_path}"
            )
            s3.s3_resource.meta.client.download_file(
                MODEL_BUCKET_NAME,
                s3_key,
                local_path,
            )

        logging.info("Model artifacts ready from S3")
    except Exception as e:
        # Fail fast so we know startup cannot proceed without artifacts
        raise MyException(e, sys)


# =====================================================
# INITIALIZE APP & ESTIMATOR
# =====================================================
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

try:
    ensure_model_artifacts_from_s3(force_download=True)
    logging.info("Initializing MovieRecommenderEstimator")
    estimator = MovieRecommenderEstimator()
except Exception as e:
    raise MyException(e, sys)


# =====================================================
# HEALTH CHECK
# =====================================================
@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "service": "movie-recommendation",
        "environment": os.getenv("ENV", "local")
    })


# =====================================================
# HOME (UI)
# =====================================================
@app.route("/")
def home():
    return render_template("index.html")


# =====================================================
# RECOMMENDATION API
# =====================================================
@app.route("/recommend", methods=["GET"])
def recommend_api():
    try:
        movie = request.args.get("title", "")
        top_n = int(request.args.get("top_n", 10))

        matched_movie, recommendations = estimator.recommend(movie, top_n)

        return jsonify({
            "matched_title": matched_movie,
            "results": recommendations.to_dict(orient="records")
        })

    except Exception as e:
        logging.error("Recommendation failed", exc_info=True)
        return jsonify({"error": str(e)}), 500


# =====================================================
# SEARCH API (Autocomplete)
# =====================================================
@app.route("/search", methods=["GET"])
def search_api():
    try:
        query = request.args.get("query", "").lower().strip()
        if not query:
            return jsonify([])

        df = estimator.recommender.df

        matches = df[
            df["title_norm"].str.contains(query, case=False, regex=False, na=False)
        ].head(10)

        return jsonify(
            matches[["title", "poster_url"]].to_dict(orient="records")
        )

    except Exception as e:
        logging.error("Search failed", exc_info=True)
        return jsonify([])


# =====================================================
# SUGGEST API (FAST TITLE SUGGESTIONS)
# =====================================================
@app.route("/suggest", methods=["GET"])
def suggest_api():
    try:
        query = request.args.get("query", "").lower().strip()
        if not query:
            return jsonify([])

        df = estimator.recommender.df

        suggestions = (
            df[df["title_norm"].str.contains(query, case=False, regex=False, na=False)]
            ["title"]
            .head(8)
            .tolist()
        )

        return jsonify(suggestions)

    except Exception as e:
        logging.error("Suggest failed", exc_info=True)
        return jsonify([])


# =====================================================
# APP RUN
# =====================================================
if __name__ == "__main__":
    print("🚀 Server running at http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000)
