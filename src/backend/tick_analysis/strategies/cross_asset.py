"""
Cross-Asset Correlation Strategies

This module implements statistical arbitrage strategies based on cross-asset
correlations, cointegration, and factor models. It includes tools for pairs
trading, basket trading, and factor-based strategies.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
from sklearn.covariance import LedoitWolf
from scipy.stats import zscore, pearsonr
from datetime import datetime, timedelta
import logging
import warnings

# Suppress specific warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', module='statsmodels.tsa.stattools')

logger = logging.getLogger(__name__)

@dataclass
class CointegrationResult:
    """Results of cointegration test between two or more time series."""
    symbol1: str
    symbol2: str
    coint_t: float
    pvalue: float
    crit_value_1pct: float
    crit_value_5pct: float
    crit_value_10pct: float
    is_cointegrated_1pct: bool
    is_cointegrated_5pct: bool
    is_cointegrated_10pct: bool
    hedge_ratio: float
    spread: Optional[np.ndarray] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'symbol1': self.symbol1,
            'symbol2': self.symbol2,
            'coint_t': self.coint_t,
            'pvalue': self.pvalue,
            'is_cointegrated_1pct': self.is_cointegrated_1pct,
            'is_cointegrated_5pct': self.is_cointegrated_5pct,
            'is_cointegrated_10pct': self.is_cointegrated_10pct,
            'hedge_ratio': self.hedge_ratio
        }

class CrossAssetStrategy:
    """
    Implements cross-asset trading strategies based on statistical relationships.
    
    This class provides functionality for:
    - Pairs trading using cointegration
    - Statistical arbitrage across correlated assets
    - Factor-based strategies
    - Risk parity portfolio construction
    """
    
    def __init__(self, 
                 lookback_period: int = 60,
                 zscore_threshold: float = 2.0,
                 min_coint_pvalue: float = 0.05,
                 min_correlation: float = 0.7):
        """
        Initialize the CrossAssetStrategy.
        
        Args:
            lookback_period: Number of periods for lookback in days
            zscore_threshold: Z-score threshold for entry/exit signals
            min_coint_pvalue: Maximum p-value for cointegration test
            min_correlation: Minimum correlation coefficient for pairs
        """
        self.lookback_period = lookback_period
        self.zscore_threshold = zscore_threshold
        self.min_coint_pvalue = min_coint_pvalue
        self.min_correlation = min_correlation
        
        # State variables
        self.price_data: Dict[str, pd.Series] = {}
        self.cointegration_pairs: List[Tuple[str, str, float]] = []
        self.correlation_matrix: pd.DataFrame = pd.DataFrame()
        self.last_updated: Optional[datetime] = None
    
    def update_prices(self, symbol: str, prices: pd.Series) -> None:
        """
        Update price data for a symbol.
        
        Args:
            symbol: Asset symbol
            prices: Time series of prices
        """
        self.price_data[symbol] = prices
        self.last_updated = datetime.utcnow()
    
    def calculate_correlations(self) -> pd.DataFrame:
        """
        Calculate correlation matrix for all assets.
        
        Returns:
            DataFrame containing correlation coefficients
        """
        if not self.price_data:
            return pd.DataFrame()
            
        # Create a DataFrame with all price series
        df = pd.DataFrame({
            sym: prices for sym, prices in self.price_data.items()
            if len(prices) >= self.lookback_period
        })
        
        if df.empty:
            return pd.DataFrame()
            
        # Calculate returns
        returns = df.pct_change().dropna()
        
        # Calculate correlation matrix
        self.correlation_matrix = returns.corr()
        return self.correlation_matrix
    
    def find_cointegrated_pairs(self, max_pairs: int = 10) -> List[CointegrationResult]:
        """
        Find cointegrated pairs of assets.
        
        Args:
            max_pairs: Maximum number of pairs to return
            
        Returns:
            List of CointegrationResult objects
        """
        if not self.price_data:
            return []
            
        symbols = list(self.price_data.keys())
        n_symbols = len(symbols)
        results = []
        
        # Calculate correlations first to filter pairs
        corr_matrix = self.calculate_correlations()
        
        for i in range(n_symbols):
            for j in range(i + 1, n_symbols):
                sym1, sym2 = symbols[i], symbols[j]
                
                # Skip if correlation is too low
                if not corr_matrix.empty and \
                   abs(corr_matrix.loc[sym1, sym2]) < self.min_correlation:
                    continue
                
                # Get price series
                prices1 = self.price_data[sym1]
                prices2 = self.price_data[sym2]
                
                # Ensure we have enough data
                if len(prices1) < self.lookback_period or len(prices2) < self.lookback_period:
                    continue
                
                # Take the most recent data points
                prices1 = prices1[-self.lookback_period:]
                prices2 = prices2[-self.lookback_period:]
                
                # Run cointegration test
                try:
                    # Test for cointegration
                    score, pvalue, _ = coint(prices1, prices2)
                    
                    # Get critical values
                    crit_values = sm.tsa.stattools.coint(prices1, prices2, autolag='AIC')[2]
                    
                    # Estimate hedge ratio
                    X = sm.add_constant(prices2)
                    model = sm.OLS(prices1, X).fit()
                    hedge_ratio = model.params[1]
                    
                    # Calculate spread
                    spread = prices1 - hedge_ratio * prices2
                    
                    # Check stationarity of spread
                    adf_result = adfuller(spread, maxlag=1)
                    
                    result = CointegrationResult(
                        symbol1=sym1,
                        symbol2=sym2,
                        coint_t=score,
                        pvalue=pvalue,
                        crit_value_1pct=crit_values[0],
                        crit_value_5pct=crit_values[1],
                        crit_value_10pct=crit_values[2],
                        is_cointegrated_1pct=score < crit_values[0],
                        is_cointegrated_5pct=score < crit_values[1],
                        is_cointegrated_10pct=score < crit_values[2],
                        hedge_ratio=hedge_ratio,
                        spread=spread.values
                    )
                    
                    # Only keep significant cointegrated pairs
                    if pvalue < self.min_coint_pvalue:
                        results.append(result)
                        
                except Exception as e:
                    logger.warning(f"Error testing {sym1}-{sym2}: {e}")
        
        # Sort by p-value and limit to max_pairs
        results.sort(key=lambda x: x.pvalue)
        return results[:max_pairs]
    
    def generate_pairs_signals(self, pair: CointegrationResult) -> Dict[str, float]:
        """
        Generate trading signals for a cointegrated pair.
        
        Args:
            pair: CointegrationResult for the pair
            
        Returns:
            Dictionary with trading signals
        """
        if pair.spread is None or len(pair.spread) < 2:
            return {}
            
        # Calculate z-score of the spread
        current_spread = pair.spread[-1]
        spread_mean = np.mean(pair.spread)
        spread_std = np.std(pair.spread)
        
        if spread_std == 0:
            return {}
            
        z_score = (current_spread - spread_mean) / spread_std
        
        # Generate signals
        signals = {
            'z_score': z_score,
            'spread': current_spread,
            'spread_mean': spread_mean,
            'spread_std': spread_std,
            'position': 0.0  # Flat by default
        }
        
        # Long signal (spread is low)
        if z_score < -self.zscore_threshold:
            signals['position'] = 1.0  # Long the spread (sell hedge_ratio of sym2, buy 1 of sym1)
            signals['signal_type'] = 'long'
        # Short signal (spread is high)
        elif z_score > self.zscore_threshold:
            signals['position'] = -1.0  # Short the spread (buy hedge_ratio of sym2, sell 1 of sym1)
            signals['signal_type'] = 'short'
        
        return signals
    
    def construct_stat_arb_portfolio(self, pairs: List[CointegrationResult]) -> Dict[str, float]:
        """
        Construct a statistical arbitrage portfolio from multiple pairs.
        
        Args:
            pairs: List of cointegrated pairs
            
        Returns:
            Dictionary of weights for each asset
        """
        if not pairs:
            return {}
            
        # Calculate signals for all pairs
        all_signals = {}
        for pair in pairs:
            signals = self.generate_pairs_signals(pair)
            if signals and 'position' in signals and signals['position'] != 0:
                all_signals[(pair.symbol1, pair.symbol2)] = signals
        
        if not all_signals:
            return {}
        
        # Calculate risk parity weights
        weights = {}
        for (sym1, sym2), signals in all_signals.items():
            position = signals['position']
            
            # Equal risk contribution (simplified)
            weight = 1.0 / len(all_signals)
            
            # Adjust weights based on position direction
            if sym1 not in weights:
                weights[sym1] = 0.0
            if sym2 not in weights:
                weights[sym2] = 0.0
                
            weights[sym1] += weight * position  # Long/short sym1
            weights[sym2] += -weight * position * signals.get('hedge_ratio', 1.0)  # Opposite for sym2
        
        # Normalize weights
        total_weight = sum(abs(w) for w in weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def calculate_risk_metrics(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate risk metrics for a given portfolio.
        
        Args:
            weights: Dictionary of asset weights
            
        Returns:
            Dictionary of risk metrics
        """
        if not weights or not self.price_data:
            return {}
        
        # Get returns for all assets with weights
        returns = {}
        for symbol in weights.keys():
            if symbol in self.price_data:
                prices = self.price_data[symbol]
                if len(prices) >= 2:
                    returns[symbol] = prices.pct_change().dropna()
        
        if not returns:
            return {}
            
        # Align returns
        returns_df = pd.DataFrame(returns)
        returns_df = returns_df.dropna()
        
        if returns_df.empty:
            return {}
        
        # Calculate portfolio returns
        weights_series = pd.Series(weights)
        portfolio_returns = (returns_df * weights_series).sum(axis=1)
        
        # Calculate risk metrics
        metrics = {
            'expected_return': portfolio_returns.mean() * 252,  # Annualized
            'volatility': portfolio_returns.std() * np.sqrt(252),  # Annualized
            'sharpe_ratio': np.sqrt(252) * portfolio_returns.mean() / portfolio_returns.std() if portfolio_returns.std() > 0 else 0,
            'max_drawdown': (portfolio_returns.cumsum().expanding().max() - portfolio_returns.cumsum()).max(),
            'skewness': portfolio_returns.skew(),
            'kurtosis': portfolio_returns.kurtosis(),
            'num_assets': len(weights)
        }
        
        return metrics
