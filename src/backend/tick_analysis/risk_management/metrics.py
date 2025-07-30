"""Risk metrics calculations for portfolio analysis."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from scipy import stats

class RiskMetrics:
    """Calculate various risk and performance metrics."""

    def __init__(
        self,
        returns: Union[pd.Series, pd.DataFrame],
        benchmark_returns: Optional[Union[pd.Series, pd.DataFrame]] = None,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
        **kwargs
    ):
        """Initialize risk metrics calculator.
        
        Args:
            returns: Historical returns data
            benchmark_returns: Benchmark returns for relative metrics
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods in a year (e.g., 252 for daily)
        """
        self.returns = returns
        self.benchmark_returns = benchmark_returns
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

    def calculate(self) -> Dict[str, float]:
        """Calculate all risk metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['total_return'] = self.total_return()
        metrics['annualized_return'] = self.annualized_return()
        metrics['volatility'] = self.volatility()
        metrics['sharpe_ratio'] = self.sharpe_ratio()
        metrics['sortino_ratio'] = self.sortino_ratio()
        metrics['max_drawdown'] = self.max_drawdown()
        
        # Relative metrics if benchmark provided
        if self.benchmark_returns is not None:
            metrics['beta'] = self.beta()
            metrics['alpha'] = self.alpha()
            metrics['information_ratio'] = self.information_ratio()
            metrics['tracking_error'] = self.tracking_error()
        
        return metrics

    def total_return(self) -> float:
        """Calculate total return."""
        if isinstance(self.returns, pd.DataFrame):
            returns = self.returns.mean(axis=1)
        else:
            returns = self.returns
        return (1 + returns).prod() - 1

    def annualized_return(self) -> float:
        """Calculate annualized return."""
        total_return = self.total_return()
        years = len(self.returns) / self.periods_per_year
        return (1 + total_return) ** (1 / years) - 1

    def volatility(self) -> float:
        """Calculate annualized volatility."""
        if isinstance(self.returns, pd.DataFrame):
            returns = self.returns.mean(axis=1)
        else:
            returns = self.returns
        return returns.std() * np.sqrt(self.periods_per_year)

    def sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio."""
        excess_return = self.annualized_return() - self.risk_free_rate
        return excess_return / self.volatility() if self.volatility() != 0 else 0

    def sortino_ratio(self) -> float:
        """Calculate Sortino ratio."""
        if isinstance(self.returns, pd.DataFrame):
            returns = self.returns.mean(axis=1)
        else:
            returns = self.returns
        
        excess_return = self.annualized_return() - self.risk_free_rate
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(self.periods_per_year)
        
        return excess_return / downside_std if downside_std != 0 else 0

    def max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        if isinstance(self.returns, pd.DataFrame):
            returns = self.returns.mean(axis=1)
        else:
            returns = self.returns
            
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = cum_returns / rolling_max - 1
        return drawdowns.min()

    def beta(self) -> float:
        """Calculate portfolio beta."""
        if self.benchmark_returns is None:
            raise ValueError("Benchmark returns required for beta calculation")
            
        if isinstance(self.returns, pd.DataFrame):
            returns = self.returns.mean(axis=1)
        else:
            returns = self.returns
            
        covariance = returns.cov(self.benchmark_returns)
        benchmark_variance = self.benchmark_returns.var()
        
        return covariance / benchmark_variance if benchmark_variance != 0 else 0

    def alpha(self) -> float:
        """Calculate portfolio alpha."""
        if self.benchmark_returns is None:
            raise ValueError("Benchmark returns required for alpha calculation")
            
        portfolio_return = self.annualized_return()
        benchmark_return = (1 + self.benchmark_returns).prod() ** (1 / (len(self.benchmark_returns) / self.periods_per_year)) - 1
        beta = self.beta()
        
        return portfolio_return - (self.risk_free_rate + beta * (benchmark_return - self.risk_free_rate))

    def information_ratio(self) -> float:
        """Calculate information ratio."""
        if self.benchmark_returns is None:
            raise ValueError("Benchmark returns required for information ratio calculation")
            
        if isinstance(self.returns, pd.DataFrame):
            returns = self.returns.mean(axis=1)
        else:
            returns = self.returns
            
        excess_returns = returns - self.benchmark_returns
        tracking_error = excess_returns.std() * np.sqrt(self.periods_per_year)
        
        return excess_returns.mean() * np.sqrt(self.periods_per_year) / tracking_error if tracking_error != 0 else 0

    def tracking_error(self) -> float:
        """Calculate tracking error."""
        if self.benchmark_returns is None:
            raise ValueError("Benchmark returns required for tracking error calculation")
            
        if isinstance(self.returns, pd.DataFrame):
            returns = self.returns.mean(axis=1)
        else:
            returns = self.returns
            
        excess_returns = returns - self.benchmark_returns
        return excess_returns.std() * np.sqrt(self.periods_per_year)

    def rolling_metrics(self, window: int = 60) -> Dict[str, pd.Series]:
        """Calculate rolling metrics."""
        if isinstance(self.returns, pd.DataFrame):
            returns = self.returns.mean(axis=1)
        else:
            returns = self.returns
            
        metrics = {}
        
        # Rolling returns
        metrics['rolling_return'] = returns.rolling(window).mean() * self.periods_per_year
        
        # Rolling volatility
        metrics['rolling_volatility'] = returns.rolling(window).std() * np.sqrt(self.periods_per_year)
        
        # Rolling Sharpe ratio
        excess_returns = returns - self.risk_free_rate / self.periods_per_year
        metrics['rolling_sharpe'] = (excess_returns.rolling(window).mean() * np.sqrt(self.periods_per_year)) / \
                                  (returns.rolling(window).std() * np.sqrt(self.periods_per_year))
        
        # Rolling beta if benchmark provided
        if self.benchmark_returns is not None:
            rolling_cov = returns.rolling(window).cov(self.benchmark_returns)
            rolling_var = self.benchmark_returns.rolling(window).var()
            metrics['rolling_beta'] = rolling_cov / rolling_var
        
        return metrics
