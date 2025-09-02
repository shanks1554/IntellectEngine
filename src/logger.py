import logging
import logger
import os

# Path to the logs directory
LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(LOGS_DIR, exist_ok = True)

# Log file path
LOG_FILE = os.path.join(LOGS_DIR, "engine.log")

def get_logger(name: str) -> logging.Logger:
    """
    Creates and returns a logger with a given name.
    All logs are stored in logs/engine.log and also shown on console.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handles if logger already exists
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        # File handler
        file_handler = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Log Format
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt = "%Y-%m-%d %H:%M:%S"
        )

        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger