# # below code is to check the logging config
import os
from src.logger import logging

from from_root import from_root

logging.debug("This is a debug message.")
logging.info("This is an info message.")
logging.warning("This is a warning message.")
logging.error("This is an error message.")
logging.critical("This is a critical message.")

# print("from_root():", from_root())

# print("cwd:", os.getcwd())