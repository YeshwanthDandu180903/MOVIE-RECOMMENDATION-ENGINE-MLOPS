import os
import sys
import pandas as pd

from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    DataIngestionArtifact,
    DataValidationArtifact
)
from src.exception import MyException
from src.logger import logging
from src.constants import TEXT_COLUMNS, COMBINED_TEXT_COLUMN


class DataTransformation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_transformation_config: DataTransformationConfig,
        data_validation_artifact: DataValidationArtifact
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
        except Exception as e:
            raise MyException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)

    def drop_unused_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop IMDb-specific numeric columns not required for content-based recommendation
        """
        drop_cols = ["imdb_rating", "imdb_voting"]
        existing_cols = [col for col in drop_cols if col in df.columns]

        logging.info(f"Dropping columns: {existing_cols}")
        return df.drop(columns=existing_cols, axis=1)

    def create_combined_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create combined_text column from multiple metadata fields
        """
        logging.info("Creating combined_text column")

        for col in TEXT_COLUMNS:
            if col not in df.columns:
                df[col] = ""

        df[COMBINED_TEXT_COLUMN] = (
            df["overview"].fillna("") + " " +
            df["genres"].fillna("") + " " +
            df["keywords"].fillna("") + " " +
            df["cast"].fillna("") + " " +
            df["director"].fillna("")
        )

        return df

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Initiates the data transformation process
        """
        logging.info("Entered Data Transformation stage")

        try:
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            df = self.read_data(
                self.data_ingestion_artifact.ingested_data_file_path
            )
            logging.info(f"Loaded data with shape: {df.shape}")

            # Step 1: Drop unwanted columns
            df = self.drop_unused_columns(df)

            # Step 2: Create combined text
            df = self.create_combined_text(df)

            # Save transformed data
            transformed_path = self.data_transformation_config.transformed_data_path
            os.makedirs(os.path.dirname(transformed_path), exist_ok=True)

            df.to_csv(transformed_path, index=False)
            logging.info(f"Transformed data saved at: {transformed_path}")

            return DataTransformationArtifact(
                transformed_data_file_path=transformed_path
            )

        except Exception as e:
            raise MyException(e, sys)
