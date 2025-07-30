"""
Logging configuration for the trading system.
"""

import os
import logging
import logging.handlers
from pathlib import Path
from typing import Optional, Dict, Any
import json
import sys
from datetime import datetime


class JsonFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_record = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "process": record.process,
            "thread": record.thread,
            "thread_name": record.threadName,
        }

        # Add exception info if present
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_record, ensure_ascii=False)


def configure_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    log_format: str = "json",
    max_bytes: int = 100 * 1024 * 1024,  # 100MB
    backup_count: int = 5,
    console_output: bool = True,
) -> None:
    """
    Configure logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file. If None, logs will only go to console.
        log_format: Log format ('json' or 'text')
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup log files to keep
        console_output: Whether to enable console output
    """
    # Default log level from environment or INFO
    log_level = log_level or os.getenv("LOG_LEVEL", "INFO").upper()

    # Create logs directory if it doesn't exist
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create formatter
    formatter = JsonFormatter() if log_format.lower() == "json" else logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Configure third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("ccxt").setLevel(logging.INFO)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("numexpr").setLevel(logging.WARNING)

    # Capture warnings
    logging.captureWarnings(True)


from typing import Optional

def get_logger(
    name: Optional[str] = None,
) -> logging.Logger:
    """
    Get a logger with the given name.

    Args:
        name: Logger name. If None, returns the root logger.

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


class LogContext:
    """Context manager for logging with additional context."""

    def __init__(self, logger: logging.Logger, **context):
        """
        Initialize with logger and context.

        Args:
            logger: Logger instance
            **context: Context variables to add to log records
        """
        self.logger = logger
        self.context = context or {}
        self.old_factory = None

    def __enter__(self):
        """Enter the context and update the log record factory."""
        self.old_factory = logging.getLogRecordFactory()

        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            for key, value in self.context.items():
                setattr(record, key, value)
            return record

        logging.setLogRecordFactory(record_factory)
        return self

    def __exit__(self, et, ev, tb):
        """Exit the context and restore the log record factory."""
        if self.old_factory:
            logging.setLogRecordFactory(self.old_factory)
