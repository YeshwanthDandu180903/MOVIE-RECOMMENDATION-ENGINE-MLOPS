import sys
from src.exception import MyException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
# from src.components.data_validation import DataValidation
# from src.components.data_transformation import DataTransformation
# from src.components.recommender_trainer import RecommenderTrainer

from src.entity.config_entity import (
    DataIngestionConfig,
    # DataValidationConfig,
    # DataTransformationConfig,
    # RecommenderModelConfig
)

from src.entity.artifact_entity import (
    DataIngestionArtifact,
    # DataValidationArtifact,
    # DataTransformationArtifact,
    # RecommenderModelArtifact
)


class TrainingPipeline:
    def __init__(self):
        try:
            self.data_ingestion_config = DataIngestionConfig()
            # self.data_validation_config = DataValidationConfig()
            # self.data_transformation_config = DataTransformationConfig()
            # self.recommender_model_config = RecommenderModelConfig()
        except Exception as e:
            raise MyException(e, sys)

    # =========================================================
    # Data Ingestion
    # =========================================================
    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Starting Data Ingestion stage")

            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config
            )
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

            logging.info(
                f"Data Ingestion completed. File stored at: "
                f"{data_ingestion_artifact.ingested_data_file_path}"
            )

            return data_ingestion_artifact

        except Exception as e:
            raise MyException(e, sys)

    # =========================================================
    # Pipeline Runner
    # =========================================================
    def run_pipeline(self) -> None:
        try:
            logging.info("===== Movie Recommendation Training Pipeline Started =====")

            data_ingestion_artifact = self.start_data_ingestion()

            # data_validation_artifact = self.start_data_validation(
            #     data_ingestion_artifact
            # )

            # data_transformation_artifact = self.start_data_transformation(
            #     data_ingestion_artifact, data_validation_artifact
            # )

            # recommender_model_artifact = self.start_recommender_trainer(
            #     data_transformation_artifact
            # )

            logging.info("===== Movie Recommendation Training Pipeline Completed =====")

        except Exception as e:
            raise MyException(e, sys)
