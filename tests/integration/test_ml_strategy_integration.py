"""
Tests for ML strategy integration with backtesting and portfolio optimization.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock, patch, ANY

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup test environment
os.environ["ENVIRONMENT"] = "testing"
os.environ["SECRET_KEY"] = "test_secret_key_123"
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["CRYPTOCOMPARE_API_KEY"] = "test_key_123"

# Mock database and other external dependencies
sys.modules["sqlalchemy"] = MagicMock()
sys.modules["sqlalchemy.orm"] = MagicMock()
sys.modules["sqlalchemy.orm.sessionmaker"] = MagicMock()

# Import modules after setting up mocks
from tick_analysis.config.settings import Settings

settings = Settings()

# Mock the logger and cache setup
with patch("tick_analysis.utils.logging_config.get_logger", return_value=MagicMock()):
    from tick_analysis.integration.ml_strategy_integration import MLStrategyIntegration
    from tick_analysis.core.strategies.ml_strategy import MLStrategy
    from tick_analysis.backtest.advanced_engine import (
        BacktestConfig,
        BacktestResult,
        AdvancedBacktestEngine,
    )
    from tick_analysis.risk.portfolio_optimizer import PortfolioOptimizer


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=100)

    data = {
        "open": 100 + np.cumsum(np.random.normal(0, 1, 100)),
        "high": 101 + np.cumsum(np.random.normal(0, 1, 100)),
        "low": 99 + np.cumsum(np.random.normal(0, 1, 100)),
        "close": 100 + np.cumsum(np.random.normal(0, 1, 100)),
        "volume": np.random.lognormal(10, 1, 100).astype(int),
    }
    return pd.DataFrame(data, index=dates)


def test_ml_strategy_integration_init():
    """Test MLStrategyIntegration initialization."""
    with patch(
        "tick_analysis.core.strategies.ml_strategy.MLStrategy.__init__", return_value=None
    ) as mock_init:
        config = {"model_type": "random_forest", "model_params": {"n_estimators": 100}}
        integration = MLStrategyIntegration(config)

        # Verify the MLStrategy was initialized with the correct parameters
        mock_init.assert_called_once_with(
            model_type="random_forest", model_params={"n_estimators": 100}, **{}
        )

        assert isinstance(integration.backtest_engine, AdvancedBacktestEngine)
        assert isinstance(integration.portfolio_optimizer, PortfolioOptimizer)
        assert integration.config == config


def test_prepare_data(sample_ohlcv_data):
    """Test data preparation."""
    integration = MLStrategyIntegration()
    features, target = integration.prepare_data(sample_ohlcv_data)

    assert not features.empty
    assert not target.empty
    assert len(features) == len(target)
    assert set(target.unique()).issubset({-1, 0, 1})


@patch.object(MLStrategy, "train")
def test_train_strategy(mock_train, sample_ohlcv_data):
    """Test strategy training."""
    mock_train.return_value = {"accuracy": 0.85}

    integration = MLStrategyIntegration()
    results = integration.train_strategy(sample_ohlcv_data)

    assert "accuracy" in results
    assert results["accuracy"] == 0.85
    mock_train.assert_called_once()


@patch.object(MLStrategy, "create_features")
@patch.object(MLStrategy, "generate_signals")
@patch.object(AdvancedBacktestEngine, "run")
def test_run_backtest(
    mock_run, mock_generate_signals, mock_create_features, sample_ohlcv_data
):
    """Test backtest execution."""
    # Setup mocks
    mock_features = pd.DataFrame(
        np.random.rand(100, 5), columns=[f"feature_{i}" for i in range(5)]
    )
    mock_signals = pd.Series(np.random.choice([-1, 0, 1], size=100))
    mock_result = BacktestResult(
        returns=pd.Series(np.random.normal(0, 0.01, 100)),
        positions=pd.DataFrame(),
        trades=pd.DataFrame(),
        portfolio_values=pd.Series(
            100000 * (1 + np.cumsum(np.random.normal(0, 0.01, 100)))
        ),
        metrics={"sharpe_ratio": 1.5, "max_drawdown": 0.1, "total_return": 0.15},
    )

    mock_create_features.return_value = mock_features
    mock_generate_signals.return_value = mock_signals
    mock_run.return_value = mock_result

    # Run test
    integration = MLStrategyIntegration()
    results = integration.run_backtest(sample_ohlcv_data)

    # Assertions
    assert "returns" in results
    assert "positions" in results
    assert "trades" in results
    assert "metrics" in results
    assert "portfolio_values" in results

    mock_create_features.assert_called_once()
    mock_generate_signals.assert_called_once()
    mock_run.assert_called_once()


@patch.object(PortfolioOptimizer, "optimize_portfolio")
def test_optimize_portfolio(mock_optimize, sample_ohlcv_data):
    """Test portfolio optimization with ML signals."""
    # Setup test data
    returns = pd.DataFrame(
        {
            "ASSET_1": np.random.normal(0.001, 0.02, 100),
            "ASSET_2": np.random.normal(0.001, 0.02, 100),
            "ASSET_3": np.random.normal(0.001, 0.02, 100),
        }
    )

    signals = pd.Series(np.random.choice([-1, 0, 1], size=100))

    # Setup mock
    mock_result = pd.DataFrame(
        {
            "asset": ["ASSET_1", "ASSET_2", "ASSET_3"],
            "weight": [0.4, 0.4, 0.2],
            "expected_return": [0.001, 0.001, 0.001],
            "volatility": [0.02, 0.02, 0.02],
        }
    )
    mock_optimize.return_value = mock_result

    # Run test
    integration = MLStrategyIntegration()
    result = integration.optimize_portfolio(returns, signals=signals)

    # Assertions
    assert not result.empty
    assert "weight" in result.columns
    assert "asset" in result.columns
    assert abs(result["weight"].sum() - 1.0) < 1e-6  # Weights should sum to 1
    mock_optimize.assert_called_once()
