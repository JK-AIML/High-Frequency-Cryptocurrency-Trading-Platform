"""
Pytest configuration and fixtures for Tick Data Analysis & Alpha Detection.

This module provides test fixtures, configuration, and setup for the entire test suite.
"""

import os
import sys
import json
from pathlib import Path
from decimal import Decimal
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Any, Optional

import pytest
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load test environment variables
load_dotenv(project_root / ".env.test")

# Set NumPy random seed for reproducible tests
np.random.seed(42)

# Test database URL (using SQLite for testing)
TEST_DATABASE_URL = "sqlite:///./data/test/test.db"

@pytest.fixture(scope="session")
def test_settings():
    """Provide test settings.

    Returns:
        Settings: Test configuration settings
    """
    from tick_analysis.config.settings import Settings

    # Override settings for testing
    settings = Settings(
        debug=True,
        log_level="DEBUG",
        test_mode=True,
        data_dir=Path("./data/test"),
        cache_dir=Path("./data/test/cache"),
        database_url=TEST_DATABASE_URL
    )

    # Create test directories if they don't exist
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.cache_dir.mkdir(parents=True, exist_ok=True)

    return settings

@pytest.fixture(scope="session")
def test_db():
    """Create and provide a test database session."""
    from tick_analysis.db.session import create_session
    from tick_analysis.db.base import Base
    
    # Create test database engine
    engine = create_session(TEST_DATABASE_URL)
    
    # Create all tables
    Base.metadata.create_all(engine)
    
    yield engine
    
    # Clean up
    Base.metadata.drop_all(engine)

# Common test data
TEST_SYMBOL = "BTC/USDT"
TEST_PRICE = Decimal("50000")
TEST_QUANTITY = Decimal("0.1")
TEST_TIMESTAMP = int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)

@pytest.fixture(scope="module")
def sample_ohlcv_data() -> pd.DataFrame:
    """Generate sample OHLCV data for testing.

    Returns:
        pd.DataFrame: DataFrame with OHLCV data and additional features
    """
    n = 1000
    dates = pd.date_range("2023-01-01", periods=n, freq="5min")
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    open_price = close - 0.1 + np.random.randn(n) * 0.2
    high = np.maximum(open_price, close) + np.random.rand(n) * 0.5
    low = np.minimum(open_price, close) - np.random.rand(n) * 0.5
    volume = np.random.lognormal(10, 1, n)

    # Create some technical indicators
    returns = close.pct_change()
    sma_20 = close.rolling(20).mean()
    rsi = 100 - (
        100
        / (
            1
            + (close.diff(1) > 0).rolling(14).mean()
            / (close.diff(1) < 0).rolling(14).mean()
        )
    )

    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "returns": returns,
            "sma_20": sma_20,
            "rsi": rsi,
        }
    )

    # Add target column
    df["target"] = (df["close"].pct_change(5).shift(-5) > 0).astype(int)

    return df.set_index("timestamp")

@pytest.fixture(scope="module")
def mock_redis():
    """Create a mock Redis client for testing."""
    from unittest.mock import MagicMock
    return MagicMock()

@pytest.fixture(scope="module")
def mock_data_collector():
    """Create a mock data collector for testing."""
    from unittest.mock import MagicMock
    collector = MagicMock()
    collector.get_ohlcv.return_value = sample_ohlcv_data()
    return collector

@pytest.fixture(scope="module")
def test_portfolio():
    """Create a test portfolio instance."""
    from tick_analysis.portfolio import Portfolio
    return Portfolio(initial_cash=Decimal("100000"))

@pytest.fixture(scope="module")
def test_strategy():
    """Create a test strategy instance."""
    from tick_analysis.strategies.volatility_strategy import VolatilityStrategy
    return VolatilityStrategy()

@pytest.fixture(scope="module")
def test_risk_manager():
    """Create a test risk manager instance."""
    from tick_analysis.risk.risk_manager import RiskManager
    return RiskManager(max_position_size=Decimal("0.1"), max_drawdown=Decimal("0.2"))

@pytest.fixture(scope="module")
def xgboost_strategy():
    """Create a test XGBoostStrategy instance."""
    from tick_analysis.strategies.ml.xgboost_strategy import XGBoostStrategy

    return XGBoostStrategy(
        name="test_strategy",
        lookback=20,
        forward_periods=5,
        model_params={
            "n_estimators": 10,  # Small for testing
            "max_depth": 3,
            "learning_rate": 0.1,
            "random_state": 42,
            "verbosity": 0,
        },
    )

# Fixtures for risk management tests
@pytest.fixture(scope="module")
def sample_returns():
    """Generate sample returns data for testing risk metrics."""
    np.random.seed(42)
    n = 1000
    dates = pd.date_range("2023-01-01", periods=n, freq="D")

    # Generate correlated returns for multiple assets
    cov_matrix = np.array(
        [[0.0004, 0.0002, 0.0001], [0.0002, 0.0009, 0.0003], [0.0001, 0.0003, 0.0016]]
    )

    # Generate correlated normal random variables
    L = np.linalg.cholesky(cov_matrix)
    uncorrelated = np.random.normal(0, 1, (n, 3))
    correlated = np.dot(uncorrelated, L.T)

    # Create DataFrame
    returns = pd.DataFrame(
        correlated, index=dates, columns=["Asset1", "Asset2", "Asset3"]
    )

    return returns

@pytest.fixture(scope="module")
def portfolio_weights():
    """Sample portfolio weights for testing."""
    return np.array([0.4, 0.3, 0.3])

@pytest.fixture(scope="module")
def risk_metrics(sample_returns):
    """Create a RiskMetrics instance for testing."""
    from tick_analysis.risk_management.metrics import RiskMetrics

    # Use the first asset as the benchmark
    returns = sample_returns["Asset1"]
    benchmark = sample_returns["Asset2"]

    return RiskMetrics(
        returns=returns,
        benchmark_returns=benchmark,
        risk_free_rate=0.05,
        periods_per_year=252,
    )

@pytest.fixture(scope="module")
def value_at_risk(sample_returns):
    """Create a ValueAtRisk instance for testing."""
    from tick_analysis.risk_management.var import ValueAtRisk

    return ValueAtRisk(
        returns=sample_returns["Asset1"], portfolio_value=10000, confidence_level=0.95
    )
