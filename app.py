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


# =====================================================
# INITIALIZE APP & ESTIMATOR
# =====================================================
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

try:
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
