import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from tick_analysis.alpha.strategies.base import BaseStrategy
from tick_analysis.alpha.strategies.ml_strategy import MLStrategy
from tick_analysis.alpha.strategies.volatility_strategy import VolatilityStrategy
from tick_analysis.alpha.strategies.volume_strategy import VolumeStrategy


class TestStrategies(unittest.TestCase):
    def setUp(self):
        # Create sample data
        np.random.seed(42)
        self.data = pd.DataFrame(
            {
                "timestamp": pd.date_range(start="2023-01-01", periods=100, freq="1D"),
                "open": np.random.normal(100, 10, 100),
                "high": np.random.normal(110, 10, 100),
                "low": np.random.normal(90, 10, 100),
                "close": np.random.normal(100, 10, 100),
                "volume": np.random.normal(1000, 200, 100),
            }
        )
        self.data.set_index("timestamp", inplace=True)

    def test_base_strategy(self):
        """Test base strategy functionality"""
        strategy = BaseStrategy("test")
        self.assertEqual(strategy.get_name(), "test")
        self.assertEqual(strategy.get_params(), {})

    @patch("tick_analysis.alpha.strategies.ml_strategy.RandomForestClassifier")
    def test_ml_strategy(self, mock_rf):
        """Test ML strategy"""
        strategy = MLStrategy()

        # Test signal generation
        signals = strategy.generate_signals(self.data)
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertIn("signal", signals.columns)

        # Test optimization
        optimized_params = strategy.optimize(self.data)
        self.assertIsInstance(optimized_params, dict)

    def test_volatility_strategy(self):
        """Test volatility strategy"""
        strategy = VolatilityStrategy()

        # Test signal generation
        signals = strategy.generate_signals(self.data)
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertIn("signal", signals.columns)
        self.assertIn("volatility", signals.columns)
        self.assertIn("z_score", signals.columns)

        # Test optimization
        optimized_params = strategy.optimize(self.data)
        self.assertIsInstance(optimized_params, dict)
        self.assertIn("window", optimized_params)

    def test_volume_strategy(self):
        """Test volume strategy"""
        strategy = VolumeStrategy()

        # Test signal generation
        signals = strategy.generate_signals(self.data)
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertIn("signal", signals.columns)
        self.assertIn("volume_z_score", signals.columns)
        self.assertIn("price_z_score", signals.columns)

        # Test optimization
        optimized_params = strategy.optimize(self.data)
        self.assertIsInstance(optimized_params, dict)
        self.assertIn("volume_window", optimized_params)
        self.assertIn("price_window", optimized_params)


class TestRiskManagement(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.returns = pd.DataFrame(
            {
                "asset1": np.random.normal(0, 0.01, 100),
                "asset2": np.random.normal(0, 0.01, 100),
                "asset3": np.random.normal(0, 0.01, 100),
            }
        )

    def test_portfolio_optimizer(self):
        """Test portfolio optimization"""
        from tick_analysis.risk.portfolio_optimizer import PortfolioOptimizer

        optimizer = PortfolioOptimizer()

        # Test optimization
        weights = optimizer.optimize_portfolio(self.returns)
        self.assertIsInstance(weights, pd.DataFrame)
        self.assertTrue(all(weights["weight"].sum() == 1))

        # Test risk metrics
        metrics = optimizer.calculate_risk_metrics(weights, self.returns)
        self.assertIn("volatility", metrics)
        self.assertIn("max_drawdown", metrics)
        self.assertIn("sharpe_ratio", metrics)

    def test_rebalancing(self):
        """Test portfolio rebalancing"""
        from tick_analysis.risk.portfolio_optimizer import PortfolioOptimizer

        optimizer = PortfolioOptimizer()

        # Create initial portfolio
        portfolio = pd.DataFrame(
            {
                "asset": ["asset1", "asset2", "asset3"],
                "weight": [0.3, 0.3, 0.4],
                "expected_return": [0.1, 0.15, 0.12],
            }
        )

        # Test rebalancing
        rebalanced = optimizer.rebalance_portfolio(portfolio, self.returns)
        self.assertIsInstance(rebalanced, pd.DataFrame)
        self.assertTrue(all(rebalanced["weight"].sum() == 1))


if __name__ == "__main__":
    unittest.main()
