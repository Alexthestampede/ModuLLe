"""
Logging configuration for ModuLLe AI providers
"""
import logging
import os
from datetime import datetime


def setup_logging(log_dir='logs', debug=False, log_level='INFO'):
    """
    Setup logging to both file and console.

    Args:
        log_dir: Directory for log files
        debug: If True, set log level to DEBUG
        log_level: Log level string (default: 'INFO')

    Returns:
        Logger instance
    """
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Set log level
    level = logging.DEBUG if debug else getattr(logging, log_level)

    # Create logger
    logger = logging.getLogger('modulle')
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    console_formatter = logging.Formatter(log_format, '%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler
    log_filename = os.path.join(log_dir, f"modulle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(level)
    file_formatter = logging.Formatter(log_format, '%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    logger.info(f"Logging initialized. Log file: {log_filename}")

    return logger


def get_logger(name):
    """
    Get a logger instance for a specific module.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(f'modulle.{name}')
