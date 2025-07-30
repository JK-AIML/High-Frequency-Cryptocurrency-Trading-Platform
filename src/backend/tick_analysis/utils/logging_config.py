"""
Logging configuration for the application.
"""

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional

def configure_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
) -> None:
    """
    Configure logging for the application.

    Args:
        log_level: The logging level to use. If None, uses LOG_LEVEL from environment or defaults to INFO.
        log_file: The path to the log file. If None, uses LOG_FILE from environment or defaults to logs/app.log.
        max_bytes: Maximum size of each log file before rotation.
        backup_count: Number of backup log files to keep.
    """
    # Get log level from environment or use default
    log_level = log_level or os.getenv("LOG_LEVEL", "INFO")
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    # Get log file path from environment or use default
    log_file = log_file or os.getenv("LOG_FILE", "logs/app.log")
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )

    # Create and configure file handler
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(numeric_level)
    root_logger.addHandler(file_handler)

    # Create and configure console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(numeric_level)
    root_logger.addHandler(console_handler)

    # Configure specific loggers
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('websockets').setLevel(logging.WARNING)

    # Log the configuration
    logging.info(f"Logging configured with level {log_level}")
    logging.info(f"Log file: {log_file}")

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger with the specified name.
    """
    return logging.getLogger(name)
