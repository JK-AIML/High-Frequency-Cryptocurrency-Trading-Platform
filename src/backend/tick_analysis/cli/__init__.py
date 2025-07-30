"""
Command-line interface for the Tick Data Analysis & Alpha Detection system.
"""

import click
from pathlib import Path
import sys
from typing import Optional


# Import commands
from .commands.download import download
from .commands.test import test
from .commands.run import run



class Context:
    """Global context object passed between CLI commands."""

    def __init__(self):
        self.debug = False
        self.config = None


# Create the main CLI group
@click.group()
@click.option("--debug/--no-debug", default=False, help="Enable debug mode.")
@click.pass_context
def cli(ctx: click.Context, debug: bool) -> None:
    """Tick Data Analysis & Alpha Detection CLI."""
    # Initialize context
    ctx.ensure_object(Context)
    ctx.obj.debug = debug

    # Configure logging
    import logging

    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    pass


# Add commands to the CLI
cli.add_command(download)
cli.add_command(test)
cli.add_command(run)


def main() -> None:
    """Entry point for the CLI."""
    try:
        cli(obj={}, auto_envvar_prefix="TDA")
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
