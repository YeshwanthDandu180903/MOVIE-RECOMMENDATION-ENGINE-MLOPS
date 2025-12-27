import os
import sys
import yaml
import pandas as pd

from src.exception import MyException
from src.logger import logging
from src.entity.artifact_entity import  DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig


class DataValidation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config

            with open(self.data_validation_config.schema_file_path, "r") as f:
                self.schema = yaml.safe_load(f)

        except Exception as e:
            raise MyException(e, sys)

    def validate_schema(self, df: pd.DataFrame) -> bool:
        """
        Check whether required columns exist
        """
        expected_columns = set(self.schema["columns"])
        actual_columns = set(df.columns)

        missing_cols = expected_columns - actual_columns
        extra_cols = actual_columns - expected_columns

        if missing_cols:
            logging.error(f"Missing columns: {missing_cols}")
        if extra_cols:
            logging.warning(f"Extra columns found: {extra_cols}")

        return len(missing_cols) == 0

    def validate_text_columns(self, df: pd.DataFrame) -> bool:
        """
        Ensure required text columns are not empty
        """
        status = True
        for col in self.schema["required_text_columns"]:
            if df[col].isnull().sum() > 0:
                logging.error(f"Null values found in column: {col}")
                status = False
        return status

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logging.info("Starting Data Validation")

            df = pd.read_csv(
                self.data_ingestion_artifact.ingested_data_file_path
            )

            schema_status = self.validate_schema(df)
            text_status = self.validate_text_columns(df)

            validation_status = schema_status and text_status

            report = {
                "schema_validation": schema_status,
                "text_column_validation": text_status,
                "overall_status": validation_status
            }

            os.makedirs(
                os.path.dirname(self.data_validation_config.validation_report_path),
                exist_ok=True
            )

            with open(self.data_validation_config.validation_report_path, "w") as f:
                yaml.dump(report, f)

            logging.info(f"Validation report saved at: "
                         f"{self.data_validation_config.validation_report_path}")

            return DataValidationArtifact(
                validation_status=validation_status,
                validation_report_file_path=self.data_validation_config.validation_report_path,
                message=None if validation_status else "Data validation failed"
            )

        except Exception as e:
            raise MyException(e, sys)
