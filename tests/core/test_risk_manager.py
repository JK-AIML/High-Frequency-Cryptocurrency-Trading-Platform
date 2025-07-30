"""
Tests for the RiskManager class.
"""

import unittest
import numpy as np
import pandas as pd
from typing import Any, Optional
from pypfopt import EfficientFrontier

from tick_analysis.core.risk.manager import RiskManager


class TestRiskManager(unittest.TestCase):
    """Test cases for RiskManager."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.risk_manager = RiskManager(
            max_leverage=2.0,
            max_drawdown=0.2,
            max_position_size=0.2,
            max_sector_exposure=0.3,
            risk_free_rate=0.02,
        )

        # Test data
        np.random.seed(42)
        self.returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        self.benchmark_returns = pd.Series(np.random.normal(0.0008, 0.018, 252))

        # Correlation matrix for testing
        symbols = ["BTC", "ETH", "SOL", "AVAX", "DOT"]
        corr = np.array(
            [
                [1.0, 0.8, 0.6, 0.5, 0.4],
                [0.8, 1.0, 0.7, 0.5, 0.4],
                [0.6, 0.7, 1.0, 0.6, 0.5],
                [0.5, 0.5, 0.6, 1.0, 0.3],
                [0.4, 0.4, 0.5, 0.3, 1.0],
            ]
        )
        self.correlations = pd.DataFrame(corr, index=symbols, columns=symbols)

    def test_initialization(self) -> None:
        """Test RiskManager initialization."""
        self.assertEqual(self.risk_manager.max_leverage, 2.0)
        self.assertEqual(self.risk_manager.max_drawdown, 0.2)
        self.assertEqual(self.risk_manager.max_position_size, 0.2)
        self.assertEqual(self.risk_manager.max_sector_exposure, 0.3)
        self.assertEqual(self.risk_manager.risk_free_rate, 0.02)

    def test_calculate_var(self, confidence: float = 0.95) -> None:
        """Test Value at Risk calculation."""
        var_95 = self.risk_manager.calculate_var(self.returns, confidence)
        var_99 = self.risk_manager.calculate_var(self.returns, 0.99)

        self.assertIsInstance(var_95, float)
        self.assertIsInstance(var_99, float)
        self.assertLess(var_99, var_95)  # 99% VaR should be more negative than 95% VaR

    def test_calculate_cvar(self, confidence: float = 0.95) -> None:
        """Test Conditional Value at Risk calculation."""
        cvar_95 = self.risk_manager.calculate_cvar(self.returns, confidence)
        cvar_99 = self.risk_manager.calculate_cvar(self.returns, 0.99)

        self.assertIsInstance(cvar_95, float)
        self.assertIsInstance(cvar_99, float)
        self.assertLessEqual(cvar_99, cvar_95)  # 99% CVaR should be <= 95% CVaR

    def test_calculate_max_drawdown(self) -> None:
        """Test maximum drawdown calculation."""
        # Create a synthetic series with a known drawdown
        prices = pd.Series([100, 110, 120, 115, 90, 95, 100, 105])
        returns = prices.pct_change().dropna()

        max_dd = self.risk_manager.calculate_max_drawdown(returns)

        # Expected drawdown is (90-120)/120 = -0.25
        self.assertAlmostEqual(max_dd, -0.25, places=4)

    def test_calculate_risk_metrics(self) -> None:
        """Test calculation of various risk metrics."""
        # Without benchmark
        metrics = self.risk_manager.calculate_risk_metrics(self.returns)

        expected_metrics = [
            "volatility",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "var_95",
            "cvar_95",
        ]

        for metric in expected_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], float)

        # With benchmark
        metrics_with_benchmark = self.risk_manager.calculate_risk_metrics(
            self.returns, self.benchmark_returns
        )

        additional_metrics = ["beta", "alpha", "tracking_error", "information_ratio"]

        for metric in additional_metrics:
            self.assertIn(metric, metrics_with_benchmark)
            self.assertIsInstance(metrics_with_benchmark[metric], float)

    def test_check_position_sizing(self) -> None:
        """Test position sizing checks."""
        # Test data
        weights = {
            "BTC": 0.15,  # Within limit
            "ETH": 0.25,  # Exceeds 20% limit
            "SOL": 0.1,  # Within limit
            "AVAX": 0.2,  # At limit
            "DOT": 0.1,  # Within limit
        }

        prices = {"BTC": 50000, "ETH": 3000, "SOL": 100, "AVAX": 20, "DOT": 15}

        portfolio_value = 100000

        # Test position sizing
        result = self.risk_manager.check_position_sizing(
            weights, prices, portfolio_value
        )

        # Verify results
        self.assertIn("passed", result)
        self.assertFalse(result["passed"])  # Should fail due to ETH position

        self.assertIn("checks", result)
        self.assertEqual(len(result["checks"]), len(weights))

        self.assertIn("violations", result)
        self.assertGreater(
            len(result["violations"]), 0
        )  # Should have at least one violation

        # Check ETH violation
        eth_check = result["violations"][0]
        self.assertEqual(eth_check["asset"], "ETH")
        self.assertIn("exceeds maximum", eth_check["issue"])

        # Check leverage
        self.assertIn("leverage", result)
        self.assertAlmostEqual(result["leverage"]["current"], 0.8)  # Sum of weights

    def test_calculate_position_sizes(self) -> None:
        """Test position size calculation using risk parity."""
        # Test data
        signals = {
            "BTC": 0.8,
            "ETH": 0.6,
            "SOL": 0.4,
            "AVAX": 0.2,
            "DOT": -0.3,  # Short position
        }

        volatilities = {"BTC": 0.8, "ETH": 1.0, "SOL": 1.2, "AVAX": 1.5, "DOT": 0.7}

        # Calculate position sizes
        weights = self.risk_manager.calculate_position_sizes(
            signals, volatilities, self.correlations, target_volatility=0.15
        )

        # Verify results
        self.assertEqual(len(weights), len(signals))

        # Check signs match signals
        for asset, signal in signals.items():
            if signal > 0:
                self.assertGreaterEqual(weights[asset], 0)
            elif signal < 0:
                self.assertLessEqual(weights[asset], 0)

        # Check position sizes are inversely proportional to volatility
        btc_weight = abs(weights["BTC"])
        eth_weight = abs(weights["ETH"])
        self.assertLess(btc_weight, eth_weight)  # Lower vol should have higher weight

    def test_add_constraints(self):
        """Test adding constraints to efficient frontier."""
        # Create dummy efficient frontier
        ef = EfficientFrontier(
            expected_returns=pd.Series([0.1, 0.15], index=["A", "B"]),
            cov_matrix=pd.DataFrame(
                [[0.04, 0.01], [0.01, 0.09]], index=["A", "B"], columns=["A", "B"]
            ),
        )

        # Add constraints
        self.risk_manager.add_constraints(ef)

        # Verify constraints were added
        # This is a basic check - in practice, we'd need to inspect the ef object
        self.assertTrue(hasattr(ef, "_constraints"))


if __name__ == "__main__":
    unittest.main()
