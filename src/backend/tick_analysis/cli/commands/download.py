"""Command for downloading market data."""

from pathlib import Path
from typing import Optional

import click

# from tick_analysis.utils.# logger import get_# logger  # Removed: # logger module does not exist

# # logger = get_# logger(__name__)


@click.command("download")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to configuration file.",
)
@click.option(
    "--symbol",
    "-s",
    required=True,
    help="Trading symbol (e.g., BTC/USD).",
)
@click.option(
    "--timeframe",
    "-t",
    default="1d",
    help="Timeframe for analysis (e.g., 1m, 1h, 1d).",
)
@click.option(
    "--start-date",
    type=click.DateTime(),
    help="Start date (YYYY-MM-DD).",
)
@click.option(
    "--end-date",
    type=click.DateTime(),
    help="End date (YYYY-MM-DD).",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    help="Output file path.",
)
@click.pass_obj
def download(
    ctx,
    config: Optional[Path],
    symbol: str,
    timeframe: str,
    start_date: Optional[str],
    end_date: Optional[str],
    output: Optional[Path],
) -> None:
    """Download market data for the given symbol and timeframe."""
    from tick_analysis.data.collector import DataCollector
    from tick_analysis.config.config.settings import Config

    # Load configuration if provided
    if config:
        cfg = Config(config)
    else:
        cfg = Config()

    # logger.info(f"Downloading {symbol} data for timeframe {timeframe}")

    try:
        collector = DataCollector()

        # Convert dates to strings if they're not None
        start_date_str = start_date if start_date else None
        end_date_str = end_date if end_date else None

        # Download the data
        data = collector.get_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date_str,
            end_date=end_date_str,
        )

        # Save or display the data
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            data.to_csv(output, index=False)
            # logger.info(f"Data saved to {output}")
        else:
            click.echo(data.to_string())

    except Exception as e:
        # logger.error(f"Failed to download data: {e}", exc_info=True)
        raise click.ClickException(f"Failed to download data: {e}")
