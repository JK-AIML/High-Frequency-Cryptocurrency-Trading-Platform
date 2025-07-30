"""
Tests for dashboard components.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import components to test
from tick_analysis.visualization.dashboard.components.market_visualization import (
    MarketVisualization,
)
from tick_analysis.visualization.dashboard.components.performance_analytics import (
    PerformanceAnalytics,
)
from tick_analysis.visualization.dashboard.components.portfolio_allocation import (
    PortfolioAllocation,
)
from tick_analysis.visualization.dashboard.components.risk_metrics import RiskMetricsDisplay


# Sample data for testing
@pytest.fixture
def sample_data():
    """Generate sample market data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")

    # Generate random walk for prices
    returns = np.random.normal(0.001, 0.02, 100)
    prices = 100 * (1 + returns.cumsum())

    # Create DataFrame with OHLCV data
    data = pd.DataFrame(
        {
            "open": prices,
            "high": prices * 1.005,
            "low": prices * 0.995,
            "close": prices * 1.002,
            "volume": np.random.lognormal(10, 1, 100) * 1000,
        },
        index=dates,
    )

    return data


@pytest.fixture
def sample_returns():
    """Generate sample returns data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")

    # Generate random returns for 3 assets
    returns = pd.DataFrame(
        {
            "BTC": np.random.normal(0.001, 0.02, 100),
            "ETH": np.random.normal(0.001, 0.02, 100) * 0.8,
            "SOL": np.random.normal(0.001, 0.02, 100) * 1.2,
        },
        index=dates,
    )

    return returns


def test_market_visualization(sample_data):
    """Test MarketVisualization component."""
    mv = MarketVisualization()

    # Test price chart rendering (no assertions, just check for exceptions)
    mv.render_price_chart(sample_data)

    # Test technical indicators
    indicators = {
        "sma_20": {"color": "red", "dash": "solid"},
        "sma_50": {"color": "blue", "dash": "dash"},
    }

    # Add indicators to data
    data_with_indicators = sample_data.copy()
    data_with_indicators["sma_20"] = data_with_indicators["close"].rolling(20).mean()
    data_with_indicators["sma_50"] = data_with_indicators["close"].rolling(50).mean()

    mv.render_technical_indicators(data_with_indicators, indicators)


def test_performance_analytics(sample_returns):
    """Test PerformanceAnalytics component."""
    pa = PerformanceAnalytics()

    # Test with single asset
    returns = sample_returns["BTC"]

    # Test metrics calculation
    metrics = pa._calculate_metrics(returns)

    assert "cagr" in metrics
    assert "sharpe" in metrics
    assert "max_drawdown" in metrics
    assert "win_rate" in metrics

    # Test with benchmark
    benchmark = sample_returns["ETH"]

    # Test equity curve rendering (no assertions, just check for exceptions)
    pa.render_equity_curve(returns, benchmark)

    # Test drawdown rendering
    pa.render_drawdown(returns)

    # Test returns distribution
    pa.render_returns_distribution(returns)

    # Test metrics display (no assertions, just check for exceptions)
    pa.render_metrics(returns)


def test_portfolio_allocation():
    """Test PortfolioAllocation component."""
    pa = PortfolioAllocation()

    # Test with sample weights
    weights = {"BTC": 0.4, "ETH": 0.35, "SOL": 0.25}

    # Test allocation chart (no assertions, just check for exceptions)
    pa.render_allocation_chart(weights)

    # Test with covariance matrix
    cov_matrix = pd.DataFrame(
        {
            "BTC": [0.04, 0.02, 0.01],
            "ETH": [0.02, 0.09, 0.03],
            "SOL": [0.01, 0.03, 0.16],
        },
        index=["BTC", "ETH", "SOL"],
    )

    # Test risk contribution (no assertions, just check for exceptions)
    pa.render_risk_contribution(weights, cov_matrix)

    # Test efficient frontier
    expected_returns = pd.Series({"BTC": 0.10, "ETH": 0.12, "SOL": 0.15})

    pa.render_efficient_frontier(
        expected_returns=expected_returns, cov_matrix=cov_matrix, risk_free_rate=0.02
    )


def test_risk_metrics(sample_returns):
    """Test RiskMetricsDisplay component."""
    rm = RiskMetricsDisplay()

    # Test with single asset
    returns = sample_returns["BTC"]

    # Test VaR/CVaR (no assertions, just check for exceptions)
    rm.render_var_cvar(returns)

    # Test drawdown analysis (no assertions, just check for exceptions)
    rm.render_drawdown_analysis(returns)

    # Test correlation heatmap with multiple assets
    rm.render_correlation_heatmap(sample_returns)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
