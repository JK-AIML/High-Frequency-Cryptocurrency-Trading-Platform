"""Value at Risk (VaR) and related risk metrics calculations."""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Union, Optional, Dict, List, Tuple

class RiskVar:
    def __init__(self) -> None:
        pass

class ValueAtRisk:
    """Calculate Value at Risk using various methods."""

    def __init__(
        self,
        returns: Union[pd.Series, pd.DataFrame],
        portfolio_value: Optional[float] = None,
        confidence_level: float = 0.95,
        **kwargs
    ):
        """Initialize VaR calculator.
        
        Args:
            returns: Historical returns data
            portfolio_value: Current portfolio value
            confidence_level: VaR confidence level (e.g., 0.95 for 95%)
        """
        self.returns = returns
        self.portfolio_value = portfolio_value or 1.0
        self.confidence_level = confidence_level

    def historical_var(self, **kwargs) -> float:
        """Calculate historical VaR."""
        if isinstance(self.returns, pd.DataFrame):
            returns = self.returns.mean(axis=1)
        else:
            returns = self.returns

        var = np.percentile(returns, (1 - self.confidence_level) * 100)
        return var * self.portfolio_value

    def parametric_var_gaussian(self, **kwargs) -> float:
        """Calculate parametric VaR using normal distribution."""
        if isinstance(self.returns, pd.DataFrame):
            returns = self.returns.mean(axis=1)
        else:
            returns = self.returns

        mean = returns.mean()
        std = returns.std()
        z_score = stats.norm.ppf(1 - self.confidence_level)
        var = mean + z_score * std
        return var * self.portfolio_value

    def parametric_var_t(self, **kwargs) -> float:
        """Calculate parametric VaR using t-distribution."""
        if isinstance(self.returns, pd.DataFrame):
            returns = self.returns.mean(axis=1)
        else:
            returns = self.returns

        mean = returns.mean()
        std = returns.std()
        df = len(returns) - 1  # degrees of freedom
        t_score = stats.t.ppf(1 - self.confidence_level, df)
        var = mean + t_score * std
        return var * self.portfolio_value

    def conditional_var(self, **kwargs) -> float:
        """Calculate conditional VaR (expected shortfall)."""
        if isinstance(self.returns, pd.DataFrame):
            returns = self.returns.mean(axis=1)
        else:
            returns = self.returns

        var = self.historical_var()
        tail_returns = returns[returns <= var / self.portfolio_value]
        cvar = tail_returns.mean()
        return cvar * self.portfolio_value

    def monte_carlo_var(self, simulations: int = 1000, **kwargs) -> float:
        """Calculate VaR using Monte Carlo simulation."""
        if isinstance(self.returns, pd.DataFrame):
            returns = self.returns.mean(axis=1)
        else:
            returns = self.returns

        mean = returns.mean()
        std = returns.std()
        simulated_returns = np.random.normal(mean, std, simulations)
        var = np.percentile(simulated_returns, (1 - self.confidence_level) * 100)
        return var * self.portfolio_value

def portfolio_var(
    returns: pd.DataFrame,
    weights: np.ndarray,
    portfolio_value: float,
    method: str = "historical",
    **kwargs
) -> float:
    """Calculate portfolio VaR.
    
    Args:
        returns: Historical returns for each asset
        weights: Portfolio weights
        portfolio_value: Current portfolio value
        method: VaR calculation method ("historical", "gaussian", "monte_carlo")
    """
    portfolio_returns = returns.dot(weights)
    var_calc = ValueAtRisk(portfolio_returns, portfolio_value, **kwargs)
    
    if method == "historical":
        return var_calc.historical_var()
    elif method == "gaussian":
        return var_calc.parametric_var_gaussian()
    elif method == "monte_carlo":
        return var_calc.monte_carlo_var(**kwargs)
    else:
        raise ValueError(f"Unsupported VaR method: {method}")

def calculate_component_var(
    returns: pd.DataFrame,
    weights: np.ndarray,
    portfolio_value: float,
    **kwargs
) -> np.ndarray:
    """Calculate component VaR for each asset."""
    # Calculate portfolio VaR
    total_var = portfolio_var(returns, weights, portfolio_value, **kwargs)
    
    # Calculate marginal VaR
    marginal_var = calculate_marginal_var(returns, weights, portfolio_value, **kwargs)
    
    # Component VaR = weight * marginal VaR
    component_var = weights * marginal_var
    
    return component_var

def calculate_incremental_var(
    returns: pd.DataFrame,
    weights: np.ndarray,
    portfolio_value: float,
    position_changes: List[float],
    **kwargs
) -> np.ndarray:
    """Calculate incremental VaR for position changes."""
    # Calculate marginal VaR
    marginal_var = calculate_marginal_var(returns, weights, portfolio_value, **kwargs)
    
    # Incremental VaR = position change * marginal VaR
    incremental_var = np.array(position_changes) * marginal_var
    
    return incremental_var

def calculate_marginal_var(
    returns: pd.DataFrame,
    weights: np.ndarray,
    portfolio_value: float,
    **kwargs
) -> np.ndarray:
    """Calculate marginal VaR for each asset."""
    # Calculate portfolio volatility
    portfolio_returns = returns.dot(weights)
    portfolio_vol = portfolio_returns.std()
    
    # Calculate asset correlations with portfolio
    correlations = returns.corrwith(portfolio_returns)
    
    # Calculate asset volatilities
    asset_vols = returns.std()
    
    # Calculate marginal VaR
    z_score = stats.norm.ppf(1 - kwargs.get('confidence_level', 0.95))
    marginal_var = -z_score * correlations * asset_vols / portfolio_vol
    
    return marginal_var
