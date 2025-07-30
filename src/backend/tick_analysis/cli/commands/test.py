"""Command for backtesting trading strategies."""

from pathlib import Path
from typing import Optional

import click

# from tick_analysis.utils.# logger import get_# logger  # Removed: # logger module does not exist

# # logger = get_# logger(__name__)


@click.command("test")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to configuration file.",
)
@click.option(
    "--strategy",
    "-s",
    required=True,
    help="Strategy to backtest (e.g., 'moving_average', 'mean_reversion').",
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
    help="Timeframe for backtest (e.g., 1m, 1h, 1d).",
)
@click.option(
    "--start-date",
    type=click.DateTime(),
    required=True,
    help="Start date for backtest (YYYY-MM-DD).",
)
@click.option(
    "--end-date",
    type=click.DateTime(),
    help="End date for backtest (YYYY-MM-DD). If not provided, uses current date.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    help="Output file path for backtest results.",
)
@click.pass_obj
def test(
    ctx,
    config: Optional[Path],
    strategy: str,
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: Optional[str],
    output: Optional[Path],
) -> None:
    """Backtest a trading strategy on historical data."""
    from tick_analysis.backtest.runner import BacktestRunner
    from tick_analysis.config.config.settings import Config

    # Load configuration if provided
    if config:
        cfg = Config(config)
    else:
        cfg = Config()

    # logger.info(f"Running backtest for {strategy} strategy on {symbol} ({timeframe})")

    try:
        # Convert dates to strings if they're not None
        start_date_str = start_date
        end_date_str = end_date if end_date else None

        # Initialize and run backtest
        runner = BacktestRunner(
            strategy_name=strategy,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date_str,
            end_date=end_date_str,
            config=cfg,
        )

        results = runner.run()

        # Save or display results
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            results.to_csv(output, index=False)
            # logger.info(f"Backtest results saved to {output}")
        else:
            # Print a summary of the results
            click.echo("\n=== Backtest Results ===")
            click.echo(f"Strategy: {strategy}")
            click.echo(f"Symbol: {symbol}")
            click.echo(f"Timeframe: {timeframe}")
            click.echo(f"Period: {start_date_str} to {end_date_str or 'now'}")
            click.echo(f"Total Return: {results.get('total_return', 'N/A')}%")
            click.echo(f"Sharpe Ratio: {results.get('sharpe_ratio', 'N/A')}")
            click.echo(f"Max Drawdown: {results.get('max_drawdown', 'N/A')}%")
            click.echo("\nUse --output to save full results to a file.")

    except Exception as e:
        # logger.error(f"Backtest failed: {e}", exc_info=True)
        raise click.ClickException(f"Backtest failed: {e}")
