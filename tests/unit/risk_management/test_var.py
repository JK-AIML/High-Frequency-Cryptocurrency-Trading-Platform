"""Unit tests for Value at Risk (VaR) and related risk metrics."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

# Import the risk management classes
from tick_analysis.risk_management.var import (
    ValueAtRisk,
    portfolio_var,
    calculate_component_var,
    calculate_incremental_var,
    calculate_marginal_var,
)
from tick_analysis.risk_management.metrics import RiskMetrics


# Test data
def create_test_returns(n=1000, seed=42):
    """Create test returns data."""
    np.random.seed(seed)
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


class TestValueAtRisk:
    """Test the ValueAtRisk class and related functions."""

    @pytest.fixture
    def test_returns(self):
        """Create test returns data."""
        return create_test_returns()

    def test_historical_var(self, test_returns):
        """Test historical VaR calculation."""
        # Test with a single asset
        var_calc = ValueAtRisk(
            test_returns["Asset1"], portfolio_value=10000, confidence_level=0.95
        )
        historical_var = var_calc.historical_var()

        # Check that VaR is negative (represents loss)
        assert historical_var < 0

        # Check that VaR is within expected range (based on test data)
        assert -300 < historical_var < 0

        # Test with a different confidence level
        var_calc_99 = ValueAtRisk(
            test_returns["Asset1"], portfolio_value=10000, confidence_level=0.99
        )
        var_99 = var_calc_99.historical_var()

        # 99% VaR should be more negative than 95% VaR (larger loss)
        assert var_99 < historical_var

    def test_parametric_var_gaussian(self, test_returns):
        """Test parametric Gaussian VaR calculation."""
        var_calc = ValueAtRisk(test_returns["Asset1"], portfolio_value=10000)
        gaussian_var = var_calc.parametric_var_gaussian()

        # Check that VaR is negative (represents loss)
        assert gaussian_var < 0

        # Check that VaR is within expected range
        assert -300 < gaussian_var < 0

    def test_parametric_var_t(self, test_returns):
        """Test parametric t-distribution VaR calculation."""
        var_calc = ValueAtRisk(test_returns["Asset1"], portfolio_value=10000)
        t_var = var_calc.parametric_var_t()

        # Check that VaR is negative (represents loss)
        assert t_var < 0

        # Check that VaR is within expected range
        assert -350 < t_var < 0

    def test_conditional_var(self, test_returns):
        """Test conditional VaR (expected shortfall) calculation."""
        var_calc = ValueAtRisk(test_returns["Asset1"], portfolio_value=10000)
        cvar = var_calc.conditional_var()

        # Calculate historical VaR for comparison
        historical_var = var_calc.historical_var()

        # CVaR should be more negative than VaR (larger loss)
        assert cvar < historical_var

    def test_monte_carlo_var(self, test_returns):
        """Test Monte Carlo VaR calculation."""
        var_calc = ValueAtRisk(test_returns["Asset1"], portfolio_value=10000)
        mc_var = var_calc.monte_carlo_var(simulations=1000)

        # Check that VaR is negative (represents loss)
        assert mc_var < 0

        # Check that VaR is within expected range
        assert -400 < mc_var < 0


class TestPortfolioRiskMetrics:
    """Test portfolio-level risk metrics and calculations."""

    @pytest.fixture
    def test_returns(self):
        """Create test returns data for multiple assets."""
        return create_test_returns()

    def test_portfolio_var(self, test_returns):
        """Test portfolio VaR calculation."""
        weights = np.array([0.4, 0.3, 0.3])

        # Test historical method
        hist_var = portfolio_var(
            returns=test_returns,
            weights=weights,
            portfolio_value=10000,
            method="historical",
        )

        # Test parametric Gaussian method
        gaussian_var = portfolio_var(
            returns=test_returns,
            weights=weights,
            portfolio_value=10000,
            method="gaussian",
        )

        # Test Monte Carlo method
        mc_var = portfolio_var(
            returns=test_returns,
            weights=weights,
            portfolio_value=10000,
            method="monte_carlo",
            simulations=1000,
        )

        # All VaR values should be negative (representing loss)
        assert hist_var < 0
        assert gaussian_var < 0
        assert mc_var < 0

        # Check that they are within reasonable bounds
        assert -1000 < hist_var < 0
        assert -1000 < gaussian_var < 0
        assert -1000 < mc_var < 0

    def test_component_var(self, test_returns):
        """Test component VaR calculation."""
        weights = np.array([0.4, 0.3, 0.3])

        # Calculate component VaR
        component_var = calculate_component_var(
            returns=test_returns, weights=weights, portfolio_value=10000
        )

        # Check that we get one component per asset
        assert len(component_var) == 3
        assert all(isinstance(x, float) for x in component_var)

        # Components should sum to the portfolio VaR (approximately)
        portfolio_var_value = portfolio_var(
            returns=test_returns, weights=weights, portfolio_value=10000
        )

        assert abs(sum(component_var) - portfolio_var_value) < 0.01 * abs(
            portfolio_var_value
        )

    def test_incremental_var(self, test_returns):
        """Test incremental VaR calculation."""
        weights = np.array([0.4, 0.3, 0.3])

        # Calculate incremental VaR for each asset
        incremental_var = calculate_incremental_var(
            returns=test_returns,
            weights=weights,
            portfolio_value=10000,
            position_changes=[0.01, 0, 0],  # Increase position in Asset1 by 1%
        )

        # Should get one value per position change
        assert len(incremental_var) == 1
        assert isinstance(incremental_var[0], float)

    def test_marginal_var(self, test_returns):
        """Test marginal VaR calculation."""
        weights = np.array([0.4, 0.3, 0.3])

        # Calculate marginal VaR for each asset
        marginal_var = calculate_marginal_var(
            returns=test_returns, weights=weights, portfolio_value=10000
        )

        # Should get one value per asset
        assert len(marginal_var) == 3
        assert all(isinstance(x, float) for x in marginal_var)

        # Marginal VaR should be negative (increasing position increases risk)
        assert all(x < 0 for x in marginal_var)


class TestRiskMetrics:
    """Test the RiskMetrics class."""

    @pytest.fixture
    def test_returns(self):
        """Create test returns data."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=252, freq="D")
        returns = pd.Series(np.random.normal(0.0005, 0.01, 252), index=dates)
        benchmark = pd.Series(np.random.normal(0.0004, 0.008, 252), index=dates)
        return returns, benchmark

    def test_initialization(self, test_returns):
        """Test RiskMetrics initialization."""
        returns, benchmark = test_returns

        # Test with benchmark
        metrics = RiskMetrics(
            returns=returns,
            benchmark_returns=benchmark,
            risk_free_rate=0.05,
            periods_per_year=252,
        )

        assert metrics.returns.equals(returns)
        assert metrics.benchmark_returns.equals(benchmark)
        assert metrics.risk_free_rate == 0.05
        assert metrics.periods_per_year == 252

        # Test without benchmark
        metrics_no_benchmark = RiskMetrics(
            returns=returns, risk_free_rate=0.05, periods_per_year=252
        )

        assert metrics_no_benchmark.benchmark_returns is None

    def test_metrics_calculation(self, test_returns):
        """Test calculation of various risk metrics."""
        returns, benchmark = test_returns
        metrics = RiskMetrics(
            returns=returns,
            benchmark_returns=benchmark,
            risk_free_rate=0.05,
            periods_per_year=252,
        )

        # Test individual metrics
        assert isinstance(metrics.annualized_return(), float)
        assert isinstance(metrics.annualized_volatility(), float)
        assert isinstance(metrics.sharpe_ratio(), float)
        assert isinstance(metrics.sortino_ratio(), float)
        assert isinstance(metrics.max_drawdown(), float)
        assert (
            0 <= metrics.max_drawdown() <= 1
        )  # Should be between 0 and 1 (0% to 100%)

        # Test with negative returns
        negative_returns = -returns
        neg_metrics = RiskMetrics(returns=negative_returns)
        assert neg_metrics.max_drawdown() > 0

        # Test beta and alpha
        beta = metrics.beta()
        alpha = metrics.alpha()

        assert isinstance(beta, float)
        assert isinstance(alpha, float)

        # Test with rolling window
        rolling_metrics = metrics.rolling_metrics(window=63)  # 3 months of daily data
        assert isinstance(rolling_metrics, pd.DataFrame)
        assert not rolling_metrics.empty

        # Test all metrics at once
        all_metrics = metrics.calculate_all_metrics()
        assert isinstance(all_metrics, dict)
        assert "sharpe_ratio" in all_metrics
        assert "max_drawdown" in all_metrics
        assert "beta" in all_metrics
        assert "alpha" in all_metrics

    def test_rolling_metrics(self, test_returns):
        """Test rolling metrics calculation."""
        returns, benchmark = test_returns
        metrics = RiskMetrics(
            returns=returns,
            benchmark_returns=benchmark,
            risk_free_rate=0.05,
            periods_per_year=252,
        )

        # Test with different windows
        for window in [21, 63, 126]:  # 1, 3, 6 months of daily data
            rolling = metrics.rolling_metrics(window=window)

            # Should have the expected number of rows
            assert len(rolling) == len(returns) - window + 1

            # Check that all metrics are calculated
            expected_columns = [
                "returns",
                "volatility",
                "sharpe_ratio",
                "max_drawdown",
                "sortino_ratio",
                "beta",
                "alpha",
                "tracking_error",
            ]

            for col in expected_columns:
                assert col in rolling.columns
                assert (
                    not rolling[col].isnull().all()
                )  # Should have some non-null values

    def test_metrics_with_benchmark(self, test_returns):
        """Test metrics that require a benchmark."""
        returns, benchmark = test_returns

        # With benchmark
        metrics = RiskMetrics(
            returns=returns,
            benchmark_returns=benchmark,
            risk_free_rate=0.05,
            periods_per_year=252,
        )

        # Without benchmark
        metrics_no_benchmark = RiskMetrics(
            returns=returns, risk_free_rate=0.05, periods_per_year=252
        )

        # Metrics that require a benchmark should return None without one
        assert metrics.beta() is not None
        assert metrics_no_benchmark.beta() is None

        assert metrics.alpha() is not None
        assert metrics_no_benchmark.alpha() is None

        assert metrics.tracking_error() is not None
        assert metrics_no_benchmark.tracking_error() is None

        assert metrics.information_ratio() is not None
        assert metrics_no_benchmark.information_ratio() is None

        # Calculate all metrics
        all_metrics = metrics.calculate_all_metrics()
        all_metrics_no_benchmark = metrics_no_benchmark.calculate_all_metrics()

        # Benchmark-relative metrics should be None
        assert all_metrics["beta"] is not None
        assert all_metrics_no_benchmark["beta"] is None

        # Other metrics should be the same
        assert all_metrics["sharpe_ratio"] == all_metrics_no_benchmark["sharpe_ratio"]
        assert all_metrics["max_drawdown"] == all_metrics_no_benchmark["max_drawdown"]

    def test_edge_cases(self):
        """Test edge cases like empty returns or NaN values."""
        # Empty returns
        empty_returns = pd.Series(dtype=float)
        with pytest.raises(ValueError):
            RiskMetrics(returns=empty_returns)

        # Single return
        single_return = pd.Series([0.01])
        metrics = RiskMetrics(returns=single_return)

        # Some metrics should work with a single return
        assert metrics.annualized_return() == 0.01 * 252  # Annualized

        # Others should return None or NaN
        assert np.isnan(metrics.annualized_volatility())
        assert metrics.sharpe_ratio() is None  # Infinite or undefined

        # Returns with NaN values
        returns_with_nan = pd.Series([0.01, np.nan, 0.02, -0.01])
        metrics = RiskMetrics(returns=returns_with_nan.dropna())

        # Should handle NaN values by dropping them
        assert len(metrics.returns) == 3  # One NaN dropped

        # All zeros
        zero_returns = pd.Series(np.zeros(100))
        metrics = RiskMetrics(returns=zero_returns)

        assert metrics.annualized_return() == 0
        assert metrics.annualized_volatility() == 0
        assert metrics.sharpe_ratio() is None  # Undefined when volatility is zero
