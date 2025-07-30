"""Command-line interface for Tick Data Analysis & Alpha Detection."""

import argparse
import logging
from pathlib import Path

from . import __version__


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Tick Data Analysis & Alpha Detection System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Show version and exit.",
    )

    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Run backtesting")
    backtest_parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default="config/backtest.yaml",
        help="Path to backtest configuration file",
    )

    # Live trading command
    live_parser = subparsers.add_parser("trade", help="Run live trading")
    live_parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default="config/live.yaml",
        help="Path to live trading configuration file",
    )

    # Data collection command
    collect_parser = subparsers.add_parser("collect", help="Collect market data")
    collect_parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default="config/collect.yaml",
        help="Path to data collection configuration file",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if not args.command:
        parser.print_help()
        return

    # Handle commands
    if args.command == "backtest":
        from .backtest.runner import run_backtest

        run_backtest(args.config)
    elif args.command == "trade":
        from .execution.runner import run_live_trading

        run_live_trading(args.config)
    elif args.command == "collect":
        from .data.runner import run_data_collection

        run_data_collection(args.config)


if __name__ == "__main__":
    main()
