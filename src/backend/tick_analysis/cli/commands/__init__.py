"""Command-line interface commands for the application."""

# This file makes the commands directory a Python package
# Import commands here to make them available when importing from tick_analysis.cli.commands
from .download import download
from .test import test
from .run import run

__all__ = ["download", "test", "run"]
