"""
Tests for the BacktestEngine class.
"""

import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import pandas as pd
import numpy as np

from tick_analysis.backtest.engine import BacktestEngine
from tick_analysis.execution.order import Order, OrderSide, OrderType, TimeInForce
from tick_analysis.portfolio import Portfolio


class TestBacktestEngine(unittest.TestCase):
    """Test cases for BacktestEngine."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Initialize test data
        self.symbol = "BTC/USDT"
        self.initial_capital = Decimal("100000")
        self.start_date = datetime(2023, 1, 1, tzinfo=timezone.utc)
        self.end_date = datetime(2023, 1, 31, tzinfo=timezone.utc)

        # Create sample price data
        dates = pd.date_range(
            start=self.start_date, end=self.end_date, freq="1h", tz="UTC"
        )

        # Generate random walk prices
        np.random.seed(42)
        log_returns = np.random.normal(0.0001, 0.01, len(dates))
        prices = np.exp(np.cumsum(log_returns)) * 50000

        self.price_data = pd.DataFrame(
            {
                "open": prices,
                "high": prices * 1.001,
                "low": prices * 0.999,
                "close": prices,
                "volume": np.random.lognormal(10, 1, len(dates)),
            },
            index=dates,
        )

        # Initialize backtest engine
        self.engine = BacktestEngine(
            initial_capital=self.initial_capital,
            data_provider=self.mock_data_provider,
            commission=Decimal("0.001"),  # 0.1%
            slippage=Decimal("0.0005"),  # 0.05%
            benchmark="BTC/USDT",
        )

    def mock_data_provider(self, symbol: str, start_date: datetime, end_date: datetime, timeframe: str = "1h") -> pd.DataFrame:
        """Mock data provider for testing."""
        # Simple data provider that returns our test data
        mask = (self.price_data.index >= start_date) & (
            self.price_data.index <= end_date
        )
        return self.price_data.loc[mask].copy()

    def test_initialization(self) -> None:
        """Test BacktestEngine initialization."""
        self.assertEqual(self.engine.initial_capital, self.initial_capital)
        self.assertEqual(self.engine.commission, Decimal("0.001"))
        self.assertEqual(self.engine.slippage, Decimal("0.0005"))
        self.assertIsNotNone(self.engine.portfolio)
        self.assertEqual(self.engine.portfolio.cash, self.initial_capital)

    def test_run_backtest(self) -> None:
        """Test running a basic backtest."""

        # Define a simple strategy that buys and holds
        def strategy(engine: BacktestEngine, data: pd.DataFrame, **kwargs) -> None:
            if not engine.portfolio.positions:
                engine.submit_order(
                    symbol=self.symbol,
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    quantity=Decimal("0.5"),
                    time_in_force=TimeInForce.GTC,
                )

        # Run backtest
        results = self.engine.run(
            strategy=strategy,
            symbols=[self.symbol],
            start_date=self.start_date,
            end_date=self.end_date,
            timeframe="1h",
        )

        # Basic assertions
        self.assertIn("returns", results)
        self.assertIn("positions", results)
        self.assertIn("trades", results)
        self.assertIn("benchmark_returns", results)

        # Check that we have at least one trade
        self.assertGreater(len(results["trades"]), 0)

        # Check final portfolio value
        self.assertGreater(self.engine.portfolio.total_value, 0)

    @patch("tick_analysis.backtest.engine.BacktestEngine._get_benchmark_data")
    def test_benchmark_comparison(self, mock_benchmark):
        """Test benchmark comparison in backtest results."""
        # Mock benchmark data
        mock_benchmark.return_value = pd.Series(
            data=np.random.normal(0.0005, 0.01, len(self.price_data)),
            index=self.price_data.index,
        )

        # Define a simple strategy
        def strategy(engine, data, **kwargs):
            pass  # Do nothing, just track benchmark

        # Run backtest
        results = self.engine.run(
            strategy=strategy,
            symbols=[self.symbol],
            start_date=self.start_date,
            end_date=self.end_date,
            timeframe="1h",
        )

        # Check benchmark returns
        self.assertIn("benchmark_returns", results)
        self.assertEqual(len(results["benchmark_returns"]), len(self.price_data))

    def test_risk_metrics(self):
        """Test calculation of risk metrics."""
        # Generate some test returns
        test_returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])

        # Calculate metrics
        metrics = self.engine.calculate_risk_metrics(test_returns)

        # Check required metrics
        required_metrics = [
            "total_return",
            "annual_return",
            "volatility",
            "sharpe_ratio",
            "max_drawdown",
            "calmar_ratio",
        ]

        for metric in required_metrics:
            self.assertIn(metric, metrics)

    def test_order_execution(self):
        """Test order execution logic."""
        # Create a market buy order
        order = Order(
            symbol=self.symbol,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            time_in_force=TimeInForce.GTC,
        )

        # Get current price data
        current_data = self.price_data.iloc[0]

        # Execute order
        trade = self.engine._execute_order(
            order=order,
            current_price=current_data["close"],
            timestamp=self.price_data.index[0],
        )

        # Check trade execution
        self.assertIsNotNone(trade)
        self.assertEqual(trade.symbol, self.symbol)
        self.assertEqual(trade.quantity, Decimal("0.1"))
        self.assertGreater(trade.price, 0)
        self.assertGreater(trade.commission, 0)


if __name__ == "__main__":
    unittest.main()
