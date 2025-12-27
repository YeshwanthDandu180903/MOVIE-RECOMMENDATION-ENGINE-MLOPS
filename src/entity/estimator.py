import sys
import pandas as pd

from src.exception import MyException
from src.logger import logging
from src.pipeline.prediction_pipeline import MovieRecommender


class MovieRecommenderEstimator:
    """
    Estimator wrapper for Movie Recommendation System.
    This replaces classifier-based estimators used in supervised ML.
    """

    def __init__(self):
        try:
            logging.info("Initializing MovieRecommenderEstimator")
            self.recommender = MovieRecommender()
        except Exception as e:
            raise MyException(e, sys)

    def recommend(self, movie_name: str, top_n: int = 10) -> pd.DataFrame:
        """
        Generate movie recommendations.

        :param movie_name: Input movie name (user query)
        :param top_n: Number of recommendations
        :return: matched movie name + recommendations dataframe
        """
        try:
            logging.info(
                f"Estimator received request: movie='{movie_name}', top_n={top_n}"
            )

            matched_movie, recommendations = self.recommender.recommend(
                movie_name=movie_name,
                top_n=top_n
            )

            return matched_movie, recommendations

        except Exception as e:
            logging.error("Error occurred in MovieRecommenderEstimator", exc_info=True)
            raise MyException(e, sys)

    def __repr__(self):
        return "MovieRecommenderEstimator()"

    def __str__(self):
        return "MovieRecommenderEstimator()"
