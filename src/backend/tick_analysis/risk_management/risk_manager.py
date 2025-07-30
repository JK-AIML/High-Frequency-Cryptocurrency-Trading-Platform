"""Risk management module for tick analysis."""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from decimal import Decimal
from datetime import datetime
from tick_analysis.api.websocket import manager as websocket_manager
import asyncio

class RiskManager:
    """Risk management class for controlling trading risk."""

    def __init__(
        self,
        max_position_size: float = 0.1,
        max_drawdown: float = 0.2,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
        max_leverage: float = 1.0,
        max_correlation: float = 0.7,
        max_sector_exposure: float = 0.3
    ):
        """Initialize risk manager.
        
        Args:
            max_position_size: Maximum position size as fraction of portfolio
            max_drawdown: Maximum allowed drawdown
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            max_leverage: Maximum allowed leverage
            max_correlation: Maximum allowed correlation between positions
            max_sector_exposure: Maximum exposure to any single sector
        """
        self.max_position_size = Decimal(str(max_position_size))
        self.max_drawdown = Decimal(str(max_drawdown))
        self.stop_loss_pct = Decimal(str(stop_loss_pct))
        self.take_profit_pct = Decimal(str(take_profit_pct))
        self.max_leverage = Decimal(str(max_leverage))
        self.max_correlation = Decimal(str(max_correlation))
        self.max_sector_exposure = Decimal(str(max_sector_exposure))
        
        self.positions: Dict[str, Dict] = {}
        self.trades: List[Dict] = []
        self.portfolio_value_history: List[float] = []
        self.sector_exposures: Dict[str, float] = {}

    def check_position_size(self, symbol: str, quantity: Decimal, price: Decimal, portfolio_value: Decimal) -> bool:
        """Check if position size is within limits.
        
        Args:
            symbol: Trading symbol
            quantity: Position quantity
            price: Current price
            portfolio_value: Total portfolio value
            
        Returns:
            bool: True if position size is acceptable
        """
        position_value = quantity * price
        position_size = position_value / portfolio_value
        
        return position_size <= self.max_position_size

    def check_drawdown(self, portfolio_value: Decimal, peak_value: Decimal) -> bool:
        """Check if current drawdown is within limits.
        
        Args:
            portfolio_value: Current portfolio value
            peak_value: Peak portfolio value
            
        Returns:
            bool: True if drawdown is acceptable
        """
        if peak_value == 0:
            return True
            
        drawdown = (peak_value - portfolio_value) / peak_value
        return drawdown <= self.max_drawdown

    def calculate_stop_loss(self, entry_price: Decimal, side: str) -> Decimal:
        """Calculate stop loss price.
        
        Args:
            entry_price: Entry price
            side: Trade side ('BUY' or 'SELL')
            
        Returns:
            Decimal: Stop loss price
        """
        if side == 'BUY':
            return entry_price * (Decimal('1') - self.stop_loss_pct)
        else:
            return entry_price * (Decimal('1') + self.stop_loss_pct)

    def calculate_take_profit(self, entry_price: Decimal, side: str) -> Decimal:
        """Calculate take profit price.
        
        Args:
            entry_price: Entry price
            side: Trade side ('BUY' or 'SELL')
            
        Returns:
            Decimal: Take profit price
        """
        if side == 'BUY':
            return entry_price * (Decimal('1') + self.take_profit_pct)
        else:
            return entry_price * (Decimal('1') - self.take_profit_pct)

    def check_leverage(self, total_exposure: Decimal, portfolio_value: Decimal) -> bool:
        """Check if leverage is within limits.
        
        Args:
            total_exposure: Total position exposure
            portfolio_value: Portfolio value
            
        Returns:
            bool: True if leverage is acceptable
        """
        leverage = total_exposure / portfolio_value
        return leverage <= self.max_leverage

    def check_correlation(self, symbol: str, returns: pd.Series, portfolio_returns: pd.Series) -> bool:
        """Check if correlation with existing positions is within limits.
        
        Args:
            symbol: Trading symbol
            returns: Asset returns series
            portfolio_returns: Portfolio returns series
            
        Returns:
            bool: True if correlation is acceptable
        """
        correlation = returns.corr(portfolio_returns)
        return abs(correlation) <= self.max_correlation

    def check_sector_exposure(self, symbol: str, sector: str, position_value: Decimal, portfolio_value: Decimal) -> bool:
        """Check if sector exposure is within limits.
        
        Args:
            symbol: Trading symbol
            sector: Sector name
            position_value: Position value
            portfolio_value: Portfolio value
            
        Returns:
            bool: True if sector exposure is acceptable
        """
        current_exposure = self.sector_exposures.get(sector, Decimal('0'))
        new_exposure = (current_exposure + position_value) / portfolio_value
        return new_exposure <= self.max_sector_exposure

    def update_position(self, symbol: str, quantity: Decimal, price: Decimal, side: str) -> None:
        """Update position information.
        
        Args:
            symbol: Trading symbol
            quantity: Position quantity
            price: Current price
            side: Trade side ('BUY' or 'SELL')
        """
        if symbol not in self.positions:
            self.positions[symbol] = {
                'quantity': Decimal('0'),
                'avg_price': Decimal('0'),
                'side': side,
                'entry_time': datetime.now()
            }
            
        position = self.positions[symbol]
        
        if side == 'BUY':
            new_quantity = position['quantity'] + quantity
            new_cost = (position['quantity'] * position['avg_price']) + (quantity * price)
            position['avg_price'] = new_cost / new_quantity
            position['quantity'] = new_quantity
        else:
            position['quantity'] -= quantity
            
        if position['quantity'] == 0:
            del self.positions[symbol]
        # WebSocket: broadcast risk update
        if hasattr(self, 'get_risk_metrics'):
            try:
                metrics = self.get_risk_metrics()
                asyncio.create_task(websocket_manager.broadcast_risk(metrics))
            except Exception:
                pass

    def add_trade(self, trade: Dict) -> None:
        """Add trade to history.
        
        Args:
            trade: Trade dictionary
        """
        self.trades.append(trade)
        self.portfolio_value_history.append(float(trade.get('portfolio_value', 0)))
        # WebSocket: broadcast risk update
        if hasattr(self, 'get_risk_metrics'):
            try:
                metrics = self.get_risk_metrics()
                asyncio.create_task(websocket_manager.broadcast_risk(metrics))
            except Exception:
                pass

    def kelly_criterion(self, win_rate: float, win_loss_ratio: float) -> float:
        """
        Calculate optimal fraction to bet using Kelly Criterion.
        Args:
            win_rate: Probability of winning (0-1)
            win_loss_ratio: Average win / average loss
        Returns:
            float: Optimal fraction of capital to risk
        """
        kelly = win_rate - (1 - win_rate) / win_loss_ratio if win_loss_ratio > 0 else 0
        return max(0, min(kelly, 1))

    def entropy_risk(self, returns: pd.Series) -> float:
        """
        Calculate entropy of return distribution (higher = more uncertainty/risk).
        Args:
            returns: Series of returns
        Returns:
            float: Entropy
        """
        hist, bin_edges = np.histogram(returns, bins=20, density=True)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log(hist))
        return float(entropy)

    def drawdown_duration(self, returns: pd.Series) -> int:
        """
        Calculate the maximum drawdown duration (number of periods underwater).
        Args:
            returns: Series of returns
        Returns:
            int: Max drawdown duration
        """
        cum_returns = (1 + returns).cumprod()
        high_water_mark = cum_returns.cummax()
        drawdown = cum_returns < high_water_mark
        durations = (drawdown != drawdown.shift()).cumsum()
        max_duration = (drawdown.groupby(durations).cumsum() * drawdown).max()
        return int(max_duration) if not np.isnan(max_duration) else 0

    def tail_risk(self, returns: pd.Series, levels=[0.01, 0.05, 0.10]) -> dict:
        """
        Calculate Conditional Value at Risk (CVaR) at multiple tail levels.
        Args:
            returns: Series of returns
            levels: List of quantile levels (e.g., [0.01, 0.05, 0.10])
        Returns:
            dict: {level: CVaR}
        """
        cvars = {}
        for level in levels:
            var = np.percentile(returns, 100 * level)
            cvar = returns[returns <= var].mean()
            cvars[f'cvar_{int(level*100)}'] = float(cvar)
        return cvars

    def detect_regime(self, returns: pd.Series) -> str:
        """
        Stub for regime detection (e.g., bull, bear, sideways).
        Args:
            returns: Series of returns
        Returns:
            str: Regime label
        """
        # Placeholder: Use rolling mean and std for simple regime detection
        mean = returns.rolling(20).mean().iloc[-1]
        std = returns.rolling(20).std().iloc[-1]
        if mean > 0.01 and std < 0.02:
            return 'bull'
        elif mean < -0.01 and std > 0.02:
            return 'bear'
        else:
            return 'sideways'

    def ml_regime_detection(self, returns: pd.Series, n_states=3) -> str:
        """
        Detect market regime using a Hidden Markov Model (HMM).
        Returns: regime label ('bull', 'bear', 'sideways', etc.)
        """
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            return 'unknown'
        if len(returns) < 100:
            return 'unknown'
        X = returns.values.reshape(-1, 1)
        model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100)
        model.fit(X)
        hidden_states = model.predict(X)
        state_means = [X[hidden_states == i].mean() for i in range(n_states)]
        regime_map = {i: 'bull' if m > 0 else 'bear' if m < 0 else 'sideways' for i, m in enumerate(state_means)}
        return regime_map[hidden_states[-1]]

    def dynamic_position_size(self, regime: str, base_size: float = 0.1) -> float:
        """
        Adjust position size based on detected regime.
        """
        if regime == 'bull':
            return base_size
        elif regime == 'sideways':
            return base_size * 0.5
        elif regime == 'bear':
            return base_size * 0.2
        else:
            return base_size * 0.1

    def stress_test(self, returns: pd.Series, scenarios: dict) -> dict:
        """
        Simulate portfolio under stress scenarios.
        scenarios: dict of {name: return_series}
        Returns: dict of {scenario: max_drawdown, min_return, recovery_time}
        """
        results = {}
        for name, scenario_returns in scenarios.items():
            cum_returns = (1 + scenario_returns).cumprod()
            max_dd = (cum_returns.cummax() - cum_returns).max() / cum_returns.cummax().max()
            min_ret = scenario_returns.min()
            # Recovery time: periods to new high after drawdown
            recovery = (cum_returns == cum_returns.cummax()).idxmax() if (cum_returns == cum_returns.cummax()).any() else None
            results[name] = {'max_drawdown': float(max_dd), 'min_return': float(min_ret), 'recovery_time': str(recovery)}
        return results

    def diversification_metrics(self, weights: np.ndarray) -> dict:
        """
        Calculate diversification metrics (e.g., HHI).
        """
        hhi = np.sum(np.square(weights))
        return {'hhi': float(hhi)}

    def get_risk_metrics(self, weights: np.ndarray = None, scenarios: dict = None) -> dict:
        """Calculate risk metrics, including advanced models."""
        if not self.portfolio_value_history:
            return {}
        returns = pd.Series(self.portfolio_value_history).pct_change().dropna()
        metrics = {
            'volatility': float(returns.std() * np.sqrt(252)),
            'sharpe_ratio': float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() != 0 else 0,
            'max_drawdown': float((pd.Series(self.portfolio_value_history).cummax() - self.portfolio_value_history).max() / pd.Series(self.portfolio_value_history).cummax()),
            'var_95': float(np.percentile(returns, 5)),
            'expected_shortfall': float(returns[returns <= np.percentile(returns, 5)].mean()),
            'win_rate': float(len(returns[returns > 0]) / len(returns)) if len(returns) > 0 else 0
        }
        # Advanced metrics
        metrics['kelly_fraction'] = self.kelly_criterion(metrics['win_rate'], abs(metrics['expected_shortfall'] / (returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 1)))
        metrics['entropy'] = self.entropy_risk(returns)
        metrics['drawdown_duration'] = self.drawdown_duration(returns)
        metrics.update(self.tail_risk(returns))
        # ML-based regime detection
        metrics['ml_regime'] = self.ml_regime_detection(returns)
        metrics['dynamic_position_size'] = self.dynamic_position_size(metrics['ml_regime'])
        # Stress testing (if scenarios provided)
        if scenarios is not None:
            metrics['stress_test'] = self.stress_test(returns, scenarios)
        # Diversification metrics (if weights provided)
        if weights is not None:
            metrics.update(self.diversification_metrics(weights))
        metrics['regime'] = self.detect_regime(returns)
        return metrics

    def should_close_position(self, symbol: str, current_price: Decimal) -> bool:
        """Check if position should be closed based on risk parameters.
        
        Args:
            symbol: Trading symbol
            current_price: Current price
            
        Returns:
            bool: True if position should be closed
        """
        if symbol not in self.positions:
            return False
            
        position = self.positions[symbol]
        pnl = (current_price - position['avg_price']) / position['avg_price']
        
        if position['side'] == 'SELL':
            pnl = -pnl
            
        return pnl <= -float(self.stop_loss_pct) or pnl >= float(self.take_profit_pct) 