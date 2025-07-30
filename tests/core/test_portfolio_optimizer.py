"""
Tests for the PortfolioOptimizer class.
"""

import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from pypfopt import EfficientFrontier, risk_models, expected_returns

from tick_analysis.core.portfolio.optimizer import PortfolioOptimizer
from tick_analysis.core.risk.manager import RiskManager


class TestPortfolioOptimizer(unittest.TestCase):
    """Test cases for PortfolioOptimizer."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.portfolio_id = 1
        self.symbols = ["BTC", "ETH", "SOL", "AVAX", "DOT"]
        self.risk_manager = RiskManager()
        self.optimizer = PortfolioOptimizer(
            portfolio_id=self.portfolio_id, risk_manager=self.risk_manager
        )

        # Mock historical data
        np.random.seed(42)
        dates = pd.date_range(start="2023-01-01", end="2023-12-31")
        self.historical_data = pd.DataFrame(
            np.random.randn(len(dates), len(self.symbols)),
            index=dates,
            columns=self.symbols,
        )

    def test_initialization(self) -> None:
        """Test PortfolioOptimizer initialization."""
        self.assertEqual(self.optimizer.portfolio_id, self.portfolio_id)
        self.assertIsInstance(self.optimizer.risk_manager, RiskManager)

    @patch("tick_analysis.core.portfolio.optimizer.PortfolioOptimizer.get_historical_data")
    def test_optimize_mean_variance(self, mock_get_data: MagicMock) -> None:
        """Test mean-variance optimization."""
        # Setup mock
        mock_get_data.return_value = self.historical_data

        # Calculate expected returns and covariance
        mu = expected_returns.mean_historical_return(self.historical_data)
        S = risk_models.sample_cov(self.historical_data)

        # Test optimization
        result = self.optimizer.optimize_mean_variance(mu, S)

        # Verify results
        self.assertIn("weights", result)
        self.assertIn("expected_return", result)
        self.assertIn("volatility", result)
        self.assertIn("sharpe_ratio", result)
        self.assertEqual(result["method"], "mean_variance")

        # Check weights sum to ~1
        self.assertAlmostEqual(sum(result["weights"].values()), 1.0, places=3)

    @patch("tick_analysis.core.portfolio.optimizer.PortfolioOptimizer.get_historical_data")
    def test_optimize_hierarchical_risk_parity(self, mock_get_data: MagicMock) -> None:
        """Test hierarchical risk parity optimization."""
        # Setup mock
        mock_get_data.return_value = self.historical_data

        # Test optimization
        result = self.optimizer.optimize_hierarchical_risk_parity(self.historical_data)

        # Verify results
        self.assertIn("weights", result)
        self.assertIn("expected_return", result)
        self.assertIn("volatility", result)
        self.assertIn("sharpe_ratio", result)
        self.assertEqual(result["method"], "hierarchical_risk_parity")

        # Check weights sum to ~1
        self.assertAlmostEqual(sum(result["weights"].values()), 1.0, places=3)

    @patch("tick_analysis.core.portfolio.optimizer.PortfolioOptimizer.get_historical_data")
    def test_optimize_portfolio(self, mock_get_data: MagicMock) -> None:
        """Test portfolio optimization with different methods."""
        # Setup mock
        mock_get_data.return_value = self.historical_data

        # Test mean-variance
        result_mv = self.optimizer.optimize_portfolio(
            self.symbols, method="mean_variance"
        )
        self.assertEqual(result_mv["method"], "mean_variance")

        # Test HRP
        result_hrp = self.optimizer.optimize_portfolio(self.symbols, method="hrp")
        self.assertEqual(result_hrp["method"], "hierarchical_risk_parity")

        # Test CVaR
        result_cvar = self.optimizer.optimize_portfolio(self.symbols, method="cvar")
        self.assertEqual(result_cvar["method"], "cvar_optimization")

        # Test invalid method
        with self.assertRaises(ValueError):
            self.optimizer.optimize_portfolio(self.symbols, method="invalid_method")

    def test_get_discrete_allocation(self) -> None:
        """Test discrete allocation calculation."""
        # Test data
        weights = {"BTC": 0.5, "ETH": 0.3, "SOL": 0.2}
        portfolio_value = 10000
        prices = pd.Series({"BTC": 50000, "ETH": 3000, "SOL": 100})

        # Test allocation
        result = self.optimizer.get_discrete_allocation(
            weights, portfolio_value, prices=prices
        )

        # Verify results
        self.assertIn("allocation", result)
        self.assertIn("leftover", result)
        self.assertIn("total_value", result)
        self.assertIn("prices", result)

        # Check allocation makes sense
        self.assertEqual(len(result["allocation"]), len(weights))
        self.assertGreaterEqual(result["leftover"], 0)
        self.assertLessEqual(result["total_value"], portfolio_value)

    def test_generate_trade_orders(self) -> None:
        """Test trade order generation."""
        # Test data
        target_weights = {"BTC": 0.5, "ETH": 0.3, "SOL": 0.2}
        current_positions = {"BTC": 0.1, "ETH": 0.4, "SOL": 0.1, "DOT": 0.4}
        portfolio_value = 10000
        prices = {"BTC": 50000, "ETH": 3000, "SOL": 100, "DOT": 20}

        # Generate orders
        orders = self.optimizer.generate_trade_orders(
            target_weights, current_positions, portfolio_value, prices
        )

        # Verify orders
        self.assertIsInstance(orders, list)

        # Check each order
        for order in orders:
            self.assertIn("symbol", order)
            self.assertIn("quantity", order)
            self.assertIn("side", order)
            self.assertIn("price", order)
            self.assertIn("value", order)
            self.assertIn(order["side"], ["buy", "sell"])
            self.assertGreater(order["quantity"], 0)
            self.assertGreater(order["value"], 0)

    @patch("tick_analysis.core.portfolio.optimizer.PortfolioOptimizer.optimize_portfolio")
    @patch("tick_analysis.core.portfolio.optimizer.PortfolioOptimizer.generate_trade_orders")
    @patch("tick_analysis.core.portfolio.optimizer.PortfolioOptimizer.get_discrete_allocation")
    def test_optimize_and_rebalance(
        self,
        mock_alloc: MagicMock,
        mock_orders: MagicMock,
        mock_optimize: MagicMock,
    ) -> None:
        """Test end-to-end optimization and rebalancing."""
        # Setup mocks
        mock_optimize.return_value = {
            "weights": {"BTC": 0.5, "ETH": 0.3, "SOL": 0.2},
            "expected_return": 0.1,
            "volatility": 0.15,
            "sharpe_ratio": 0.8,
            "method": "mean_variance",
        }

        mock_orders.return_value = [
            {
                "symbol": "BTC",
                "side": "buy",
                "quantity": 0.1,
                "price": 50000,
                "value": 5000,
            },
            {
                "symbol": "ETH",
                "side": "sell",
                "quantity": 0.05,
                "price": 3000,
                "value": 150,
            },
        ]

        mock_alloc.return_value = {
            "allocation": {"BTC": 0.5, "ETH": 0.3, "SOL": 0.2},
            "leftover": 100,
            "total_value": 9900,
            "prices": {"BTC": 50000, "ETH": 3000, "SOL": 100},
        }

        # Test rebalancing
        result = self.optimizer.optimize_and_rebalance(
            symbols=self.symbols,
            method="mean_variance",
            portfolio_value=10000,
            current_positions={"BTC": 0.4, "ETH": 0.35, "SOL": 0.15, "DOT": 0.1},
        )

        # Verify results
        self.assertIn("optimization", result)
        self.assertIn("allocation", result)
        self.assertIn("orders", result)
        self.assertIn("prices", result)
        self.assertIn("current_positions", result)

        # Verify mocks were called correctly
        mock_optimize.assert_called_once_with(
            self.symbols,
            method="mean_variance",
            portfolio_value=10000,
            current_positions={"BTC": 0.4, "ETH": 0.35, "SOL": 0.15, "DOT": 0.1},
        )
        mock_orders.assert_called_once()
        mock_alloc.assert_called_once()


if __name__ == "__main__":
    unittest.main()
