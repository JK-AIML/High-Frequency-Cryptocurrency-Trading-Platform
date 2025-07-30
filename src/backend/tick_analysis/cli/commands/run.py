"""Command for running the Streamlit dashboard."""

import subprocess
import sys
import webbrowser
import time
from pathlib import Path
from typing import Optional

import click

# from tick_analysis.utils.# logger import get_# logger  # Removed: # logger module does not exist

# # logger = get_# logger(__name__)


@click.command("run")
@click.option(
    "--host",
    default="0.0.0.0",
    help="The address to bind the server to.",
    show_default=True,
)
@click.option(
    "--port",
    default=8501,
    type=int,
    help="The port to run the server on.",
    show_default=True,
)
@click.option(
    "--no-browser",
    is_flag=True,
    help="Don't open the app in a browser after starting the server.",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to configuration file.",
)
@click.pass_obj
def run(
    ctx,
    host: str,
    port: int,
    no_browser: bool,
    config: Optional[Path],
) -> None:
    """Run the Streamlit dashboard."""
    from tick_analysis.config.config.settings import Config

    # Load configuration if provided
    if config:
        Config(config)  # This will be available as a singleton

    # Build the Streamlit command
    cmd = [
        sys.executable,  # Use the same Python interpreter
        "-m",
        "streamlit",
        "run",
        "app.py",
        "--server.port",
        str(port),
        "--server.address",
        host,
        "--server.headless",
        "true",
        "--# logger.level",
        "debug" if ctx.debug else "info",
    ]

    # Open browser after a short delay if not disabled
    if not no_browser:
        url = f"http://{host}:{port}"

        def open_browser():
            time.sleep(2)  # Give the server a moment to start
            try:
                webbrowser.open_new_tab(url)
            except Exception as e:
                pass

        import threading

        threading.Thread(target=open_browser, daemon=True).start()

    try:
        # logger.info(f"Starting Streamlit server on {host}:{port}")
        # logger.info("Press Ctrl+C to stop the server")
        # Run the Streamlit server
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        sys.exit(1)
