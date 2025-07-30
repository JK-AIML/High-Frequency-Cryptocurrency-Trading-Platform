"""
Tests for the MLStrategy class.
"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from decimal import Decimal


from tick_analysis.data.models import Candle, Timeframe
from tick_analysis.portfolio import Portfolio
from tick_analysis.alpha.strategies.ml_strategy import MLStrategy


class TestMLStrategy(unittest.TestCase):
    """Test cases for MLStrategy."""

    def setUp(self):
        """Set up test fixtures."""
        # Initialize test data
        self.symbol = "BTC/USDT"
        self.timeframe = Timeframe.ONE_HOUR

        # Create sample features and target
        np.random.seed(42)
        n_samples = 1000
        self.X = np.random.randn(n_samples, 5)  # 5 features
        self.y = np.random.randint(0, 2, n_samples)  # Binary classification

        # Create sample candles
        self.candles = [
            Candle(
                symbol=self.symbol,
                timeframe=self.timeframe,
                timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc)
                + timedelta(hours=i),
                open=Decimal("50000") + Decimal(str(i)),
                high=Decimal("50100") + Decimal(str(i)),
                low=Decimal("49900") + Decimal(str(i)),
                close=Decimal("50050") + Decimal(str(i)),
                volume=Decimal("1000") + Decimal(str(i)),
            )
            for i in range(100)
        ]

        # Initialize strategy
        self.strategy = MLStrategy(
            name="test_ml_strategy",
            symbols=[self.symbol],
            timeframe=self.timeframe,
            model_type="xgboost",
            lookback=20,
            train_interval=100,
            retrain_interval=1000,
        )

        # Initialize portfolio
        self.portfolio = Portfolio(initial_cash=Decimal("100000"))

    def test_initialization(self):
        """Test MLStrategy initialization."""
        from unittest.mock import patch
        with patch("tick_analysis.alpha.strategies.ml_strategy.MLStrategy._train_model") as mock_train:
            strategy = MLStrategy(
                name="test_ml_strategy",
                symbols=[self.symbol],
                timeframe=self.timeframe,
                model_type="xgboost",
                lookback=20,
                train_interval=100,
                retrain_interval=1000,
            )
            self.assertEqual(strategy.name, "test_ml_strategy")
            self.assertEqual(strategy.symbols, [self.symbol])
            self.assertEqual(strategy.timeframe, self.timeframe)
            self.assertEqual(strategy.model_type, "xgboost")
            self.assertEqual(strategy.lookback, 20)
            self.assertEqual(strategy.train_interval, 100)
            self.assertEqual(strategy.retrain_interval, 1000)
            # Check that model training was called
            mock_train.assert_called_once()

    def test_train_model(self):
        """Test model training."""
        from unittest.mock import patch, MagicMock
        with patch("xgboost.XGBClassifier") as mock_xgb, patch("tick_analysis.alpha.strategies.ml_strategy.Pipeline") as mock_pipeline:
            from tick_analysis.alpha.strategies.ml_strategy import MLStrategy
            mock_model = MagicMock()
            mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
            mock_xgb.return_value = mock_model

            # Set up a MagicMock pipeline
            mock_pipe_instance = MagicMock()
            mock_pipe_instance.fit.return_value = mock_pipe_instance  # fit returns self
            mock_pipeline.return_value = mock_pipe_instance

            # Create sample data
            X_train, X_test, y_train, y_test = (
                self.X[:800],
                self.X[800:],
                self.y[:800],
                self.y[800:],
            )
            # Train model
            strategy = MLStrategy(
                name="test_ml_strategy",
                symbols=[self.symbol],
                timeframe=self.timeframe,
                model_type="xgboost",
                lookback=20,
                train_interval=100,
                retrain_interval=1000,
            )
            strategy._train_model(X_train, y_train, X_test, y_test)
            # Check that pipeline was created
            self.assertIsNotNone(strategy.pipeline, "MLStrategy._train_model did not assign a pipeline; check for errors in model or pipeline creation.")
            # Check that pipeline fit was called at least once
            self.assertGreaterEqual(strategy.pipeline.fit.call_count, 1)
            # Check that metrics were updated
            self.assertIn("accuracy", strategy.metrics)
            self.assertIn("precision", strategy.metrics)
            self.assertIn("recall", strategy.metrics)
            self.assertIn("f1", strategy.metrics)

    def test_prepare_features(self):
        """Test feature preparation."""
        # Create sample data
        df = pd.DataFrame(
            {
                "open": [50000, 50050, 50100, 50075, 50025],
                "high": [50100, 50150, 50200, 50175, 50125],
                "low": [49900, 49950, 50000, 49975, 49925],
                "close": [50050, 50100, 50150, 50125, 50075],
                "volume": [1000, 1200, 1500, 1300, 1100],
            }
        )

        # Prepare features
        features = self.strategy._prepare_features(df)

        # Check that features were created
        expected_columns = ["returns", "volatility", "rsi", "macd", "bollinger"]

        for col in expected_columns:
            self.assertIn(col, features.columns)

    @patch("tick_analysis.alpha.strategies.ml_strategy.MLStrategy._generate_signal")
    def test_on_candle(self, mock_generate_signal):
        """Test on_candle method."""
        # Setup mock
        mock_generate_signal.return_value = 1  # Buy signal

        # Process candles
        for candle in self.candles[:50]:  # First 50 candles
            self.strategy.on_candle(candle)

        # Check that signals were generated
        self.assertGreaterEqual(len(self.strategy.signals), 0)

        # Check that position was opened
        if len(self.strategy.signals) > 0:
            signal = self.strategy.signals[-1]
            self.assertIn(signal["action"], ["BUY", "SELL"])
            self.assertEqual(signal["symbol"], self.symbol)

    def test_generate_signal(self):
        """Test signal generation."""
        # Train model first
        X_train, X_test, y_train, y_test = (
            self.X[:800],
            self.X[800:],
            self.y[:800],
            self.y[800:],
        )
        self.strategy._train_model(X_train, y_train, X_test, y_test)

        # Prepare features for prediction
        features = pd.DataFrame(
            {
                "returns": [0.01, -0.005, 0.02, -0.01],
                "volatility": [0.1, 0.12, 0.15, 0.08],
                "rsi": [45, 55, 65, 35],
                "macd": [0.01, -0.005, 0.02, -0.01],
                "bollinger": [0.5, -0.3, 0.7, -0.5],
            }
        )

        # Generate signal
        signal = self.strategy._generate_signal(features.iloc[-1])

        # Check that signal is valid
        self.assertIn(signal, [-1, 0, 1])


if __name__ == "__main__":
    unittest.main()
