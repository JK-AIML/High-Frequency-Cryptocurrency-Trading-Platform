"""
Configuration package for the trading system.
"""

from .settings import *
from .logging_config import configure_logging

# Configure logging when the config package is imported
configure_logging()
