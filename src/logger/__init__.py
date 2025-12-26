# filepath: C:\Users\abhis\yeshwanth\MOVIE-RECOMMENDATION-ENGINE-MLOPS\src\logger\__init__.py
import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from pathlib import Path

try:  # Add this line
    # Constants for log configuration
    LOG_DIR = 'logs'
    LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
    MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
    BACKUP_COUNT = 3  # Number of backup log files to keep

    # Determine the project root directory using pathlib
    # Get the directory of the current file (__init__.py)
    current_file_path = Path(__file__).resolve()
    # Go up two levels to reach the project root
    PROJECT_ROOT_DIR = current_file_path.parent.parent.parent
    # Construct the log directory path
    log_dir_path = PROJECT_ROOT_DIR / LOG_DIR

    # Create the log directory if it doesn't exist
    log_dir_path.mkdir(parents=True, exist_ok=True)  # Use pathlib's mkdir

    # Construct the log file path
    log_file_path = log_dir_path / LOG_FILE

    def configure_logger():
        """
        Configures logging with a rotating file handler and a console handler.
        """
        # Create a custom logger
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

        # Define formatter
        formatter = logging.Formatter("[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s")

        # File handler with rotation
        file_handler = RotatingFileHandler(log_file_path, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)

        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    # Configure the logger
    configure_logger()
except Exception as e:  # Add this line
    print(f"An error occurred: {e}")  # Add this line