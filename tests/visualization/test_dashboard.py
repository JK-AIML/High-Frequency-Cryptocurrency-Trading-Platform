"""
Tests for the Dashboard class.
"""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timezone, timedelta

from tick_analysis.visualization.dashboard import Dashboard
from tick_analysis.data.models import Candle, Timeframe


class TestDashboard(unittest.TestCase):
    """Test cases for Dashboard."""

    def setUp(self):
        """Set up test fixtures."""
        # Initialize test data
        self.symbol = "BTC/USDT"
        self.timeframe = Timeframe.ONE_HOUR
        self.start_date = datetime(2023, 1, 1, tzinfo=timezone.utc)
        self.end_date = datetime(2023, 1, 31, tzinfo=timezone.utc)

        # Generate sample OHLCV data
        dates = pd.date_range(
            start=self.start_date, end=self.end_date, freq="1h", tz="UTC"
        )

        np.random.seed(42)
        log_returns = np.random.normal(0.0001, 0.01, len(dates))
        prices = np.exp(np.cumsum(log_returns)) * 50000

        self.ohlcv_data = pd.DataFrame(
            {
                "open": prices,
                "high": prices * 1.001,
                "low": prices * 0.999,
                "close": prices,
                "volume": np.random.lognormal(10, 1, len(dates)),
            },
            index=dates,
        )

        # Generate sample trades
        self.trades = [
            {
                "timestamp": self.start_date + timedelta(hours=i * 24),
                "symbol": self.symbol,
                "side": "BUY" if i % 2 == 0 else "SELL",
                "price": float(prices[i * 24]),
                "quantity": 0.1,
                "pnl": np.random.normal(10, 5),
            }
            for i in range(10)
        ]

        # Generate sample performance metrics
        self.metrics = {
            "total_return": 0.15,
            "annual_return": 0.25,
            "volatility": 0.12,
            "sharpe_ratio": 1.8,
            "max_drawdown": -0.08,
            "win_rate": 0.65,
            "profit_factor": 1.5,
            "trades": 100,
        }

        # Initialize dashboard
        self.dashboard = Dashboard()

    def test_initialization(self):
        """Test Dashboard initialization."""
        self.assertIsInstance(self.dashboard.fig, go.Figure)
        self.assertEqual(len(self.dashboard.fig.data), 0)

    def test_add_candlestick_chart(self):
        """Test adding candlestick chart."""
        # Add candlestick chart
        self.dashboard.add_candlestick_chart(df=self.ohlcv_data, name=self.symbol)

        # Check that figure was updated
        self.assertEqual(len(self.dashboard.fig.data), 1)
        self.assertEqual(self.dashboard.fig.data[0].type, "candlestick")

        # Check layout updates
        self.assertIsNotNone(self.dashboard.fig.layout.xaxis)
        self.assertIsNotNone(self.dashboard.fig.layout.yaxis)

    def test_add_volume_chart(self):
        """Test adding volume chart."""
        # Add volume chart
        self.dashboard.add_volume_chart(
            df=self.ohlcv_data, name=f"{self.symbol} Volume"
        )

        # Check that figure was updated
        self.assertEqual(len(self.dashboard.fig.data), 1)
        self.assertEqual(self.dashboard.fig.data[0].type, "bar")

    def test_add_trade_markers(self):
        """Test adding trade markers to chart."""
        # Add candlestick chart first
        self.dashboard.add_candlestick_chart(df=self.ohlcv_data, name=self.symbol)

        # Add trade markers
        self.dashboard.add_trade_markers(
            trades=self.trades, price_data=self.ohlcv_data["close"]
        )

        # Check that markers were added
        # Should have candlesticks (1) + buy markers + sell markers
        self.assertGreater(len(self.dashboard.fig.data), 1)

    def test_add_technical_indicators(self):
        """Test adding technical indicators."""
        # Add candlestick chart first
        self.dashboard.add_candlestick_chart(df=self.ohlcv_data, name=self.symbol)

        # Add technical indicators
        self.dashboard.add_technical_indicators(
            df=self.ohlcv_data, indicators=["sma", "rsi", "macd"]
        )

        # Check that indicators were added
        # Should have candlesticks + SMA + RSI + MACD (3)
        self.assertEqual(len(self.dashboard.fig.data), 4)

    def test_add_performance_metrics(self):
        """Test adding performance metrics table."""
        # Add metrics table
        self.dashboard.add_performance_metrics(
            metrics=self.metrics, title="Strategy Performance"
        )

        # Check that annotations were added
        self.assertIsNotNone(self.dashboard.fig.layout.annotations)
        self.assertGreater(len(self.dashboard.fig.layout.annotations), 0)

    @patch("plotly.graph_objects.Figure.show")
    def test_show(self, mock_show):
        """Test showing the dashboard."""
        # Add some content
        self.dashboard.add_candlestick_chart(df=self.ohlcv_data, name=self.symbol)

        # Show dashboard
        self.dashboard.show()

        # Check that show was called
        mock_show.assert_called_once()


if __name__ == "__main__":
    unittest.main()
