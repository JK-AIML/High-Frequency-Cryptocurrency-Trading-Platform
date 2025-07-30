"""
Main entry point for the Crypto Portfolio Optimization & Trading System.
"""

import asyncio
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from prometheus_client import Counter, Gauge, Histogram, start_http_server

from tick_analysis.backtest.advanced_engine import (
    AdvancedBacktestEngine,
    BacktestConfig,
    BacktestResult,
)
from tick_analysis.models import Portfolio, Timeframe
from tick_analysis.data.collectors.cryptocompare_collector import CryptoCompareCollector
from tick_analysis.execution.executor import ExecutionHandler
from tick_analysis.portfolio.optimizer import (
    MeanVarianceOptimizer,
    RiskParityOptimizer,
    HierarchicalRiskParityOptimizer,
)
from tick_analysis.risk.advanced_manager import AdvancedRiskManager, RiskParameters
from tick_analysis.strategies.moving_average_crossover import MovingAverageCrossover
from tick_analysis.strategies.mean_reversion import MeanReversionStrategy
from tick_analysis.utils.logging_config import configure_logging

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Prometheus metrics
TRADING_SIGNALS = Counter('trading_signals_total', 'Total number of trading signals generated', ['strategy', 'symbol'])
POSITION_SIZE = Gauge('position_size', 'Current position size', ['symbol'])
PORTFOLIO_VALUE = Gauge('portfolio_value', 'Current portfolio value')
RISK_METRICS = Gauge('risk_metrics', 'Current risk metrics', ['metric'])
API_LATENCY = Histogram('api_latency_seconds', 'API request latency', ['endpoint'])
ERROR_COUNTER = Counter('error_total', 'Total number of errors', ['type'])

class TradingSystem:
    """Main trading system that coordinates all components."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the trading system."""
        self.config = config or {}
        self.running = False
        self.portfolio = Portfolio(cash=self.config.get("initial_capital", 100000.0))

        # Initialize components
        self.data_collector = CryptoCompareCollector()
        self.risk_manager = AdvancedRiskManager()
        self.execution_handler = ExecutionHandler()

        # Initialize strategies
        self.strategies = {
            "moving_average": MovingAverageCrossover(),
            "mean_reversion": MeanReversionStrategy(),
        }

        # Initialize optimizers
        self.optimizers = {
            "mean_variance": MeanVarianceOptimizer(),
            "risk_parity": RiskParityOptimizer(),
            "hierarchical_risk_parity": HierarchicalRiskParityOptimizer(),
        }

        # Backtesting engine
        self.backtest_engine = AdvancedBacktestEngine()

        # Register signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        # Start Prometheus metrics server
        start_http_server(8000)

    async def start(self) -> None:
        """Start the trading system."""
        if self.running:
            logger.warning("Trading system is already running")
            return

        logger.info("Starting trading system...")
        self.running = True

        try:
            # Initialize components
            await self.data_collector.initialize()

            # Main event loop
            while self.running:
                try:
                    start_time = time.time()

                    # Update market data
                    await self._update_market_data()

                    # Generate signals from strategies
                    signals = await self._generate_signals()

                    # Generate and execute orders
                    await self._execute_strategy(signals)

                    # Run portfolio optimization
                    await self._optimize_portfolio()

                    # Monitor risk
                    await self._monitor_risk()

                    # Update metrics
                    self._update_metrics()

                    # Sleep to avoid excessive API calls
                    await asyncio.sleep(60)  # 1 minute

                except Exception as e:
                    ERROR_COUNTER.labels(type='main_loop').inc()
                    logger.error(f"Error in main loop: {e}", exc_info=True)
                    await asyncio.sleep(60)  # Wait before retrying

        except asyncio.CancelledError:
            logger.info("Trading system shutdown requested")
        except Exception as e:
            ERROR_COUNTER.labels(type='fatal').inc()
            logger.critical(f"Fatal error in trading system: {e}", exc_info=True)
        finally:
            await self.shutdown()

    def _update_metrics(self) -> None:
        """Update Prometheus metrics."""
        try:
            # Update portfolio value
            PORTFOLIO_VALUE.set(self.portfolio.total_value)

            # Update position sizes
            for symbol, position in self.portfolio.positions.items():
                POSITION_SIZE.labels(symbol=symbol).set(position.size)

            # Update risk metrics
            risk_metrics = self.risk_manager.get_metrics()
            for metric, value in risk_metrics.items():
                RISK_METRICS.labels(metric=metric).set(value)

        except Exception as e:
            ERROR_COUNTER.labels(type='metrics_update').inc()
            logger.error(f"Error updating metrics: {e}", exc_info=True)

    async def _update_market_data(self) -> None:
        """Update market data for all symbols."""
        try:
            with API_LATENCY.labels(endpoint='market_data').time():
                # Get list of symbols we're interested in
                symbols = self.config.get("symbols", ["BTC", "ETH", "SOL", "BNB", "XRP"])

                # Fetch OHLCV data
                for symbol in symbols:
                    ohlcv = await self.data_collector.get_ohlcv(
                        symbol=symbol,
                        timeframe=Timeframe.DAILY,
                        limit=100,  # Last 100 candles
                    )

                    # Store data for strategies
                    logger.debug(f"Updated data for {symbol}: {len(ohlcv)} candles")

        except Exception as e:
            ERROR_COUNTER.labels(type='market_data').inc()
            logger.error(f"Error updating market data: {e}", exc_info=True)

    async def _fetch_data(self) -> Dict[str, Any]:
        # Minimal stub for mypy compliance
        return {}

    async def _generate_signals(self) -> Dict[str, Dict[str, float]]:
        """Generate trading signals from all strategies."""
        signals = {}

        try:
            # Get historical data for strategies
            historical_data: Dict[str, Any] = await self._fetch_data()

            for strategy_name, strategy in self.strategies.items():
                try:
                    # Generate signals using the strategy
                    strategy_signals = await strategy.generate_signals(historical_data)
                    signals[strategy_name] = strategy_signals

                    # Update metrics
                    for symbol, signal in strategy_signals.items():
                        TRADING_SIGNALS.labels(strategy=strategy_name, symbol=symbol).inc()

                except Exception as e:
                    ERROR_COUNTER.labels(type='signal_generation').inc()
                    logger.error(f"Error generating signals with {strategy_name}: {e}", exc_info=True)

        except Exception as e:
            ERROR_COUNTER.labels(type='signal_generation').inc()
            logger.error(f"Error in signal generation: {e}", exc_info=True)

        return signals

    async def _execute_strategy(self, signals: Dict[str, Dict[str, float]]) -> None:
        """Execute trades based on strategy signals."""
        # Aggregate signals from all strategies
        aggregated_signals = await self._aggregate_signals(signals)

        # Generate and execute orders
        for symbol, signal_strength in aggregated_signals.items():
            try:
                # Skip if signal is neutral
                if signal_strength == 0:
                    continue

                # Get current price
                current_price = await self._get_current_price(symbol)
                if current_price is None:
                    continue

                # Calculate position size using risk management
                stop_loss = self._calculate_stop_loss(
                    symbol, current_price, signal_strength
                )
                position_size = await self._calculate_position_size(
                    symbol=symbol,
                    price=current_price,
                    stop_loss=stop_loss,
                    signal_strength=signal_strength,
                )

                if position_size <= 0:
                    continue

                # Create and submit order
                side = "buy" if signal_strength > 0 else "sell"
                order = {
                    "symbol": symbol,
                    "side": side,
                    "quantity": position_size,
                    "price": current_price,
                    "stop_loss": stop_loss,
                    "take_profit": self._calculate_take_profit(
                        current_price, stop_loss
                    ),
                }

                # Execute order
                await self.execution_handler.execute_order(order)

            except Exception as e:
                logger.error(
                    f"Error executing strategy for {symbol}: {e}", exc_info=True
                )

    async def _optimize_portfolio(self) -> None:
        """Run portfolio optimization."""
        try:
            # Get historical returns and covariance matrix
            returns, cov_matrix = await self._get_returns_data()

            if returns is None or cov_matrix is None:
                return

            # Select optimizer based on config
            optimizer_type = self.config.get("optimizer", "risk_parity")
            optimizer = self.optimizers.get(optimizer_type)

            if optimizer is None:
                logger.warning(f"Unknown optimizer: {optimizer_type}")
                return

            # Run optimization
            result = optimizer.optimize(returns=returns, cov_matrix=cov_matrix)

            # Rebalance portfolio based on optimization result
            await self._rebalance_portfolio(result.weights)

        except Exception as e:
            logger.error(f"Error in portfolio optimization: {e}", exc_info=True)

    async def _monitor_risk(self) -> None:
        """Monitor portfolio risk and take action if needed."""
        try:
            # Generate risk report
            risk_report = self.risk_manager.generate_risk_report(self.portfolio)

            # Check if we're within risk limits
            within_limits, message = self.risk_manager.check_risk_limits(self.portfolio)

            if not within_limits:
                logger.warning(f"Risk limit exceeded: {message}")
                await self._handle_risk_limit_breach()

            # Log risk metrics periodically
            logger.info(f"Portfolio risk metrics: {risk_report.get('metrics', {})}")

        except Exception as e:
            logger.error(f"Error in risk monitoring: {e}", exc_info=True)

    async def _aggregate_signals(
        self, signals: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Aggregate signals from multiple strategies."""
        # Simple majority voting for now
        aggregated = {}

        for strategy_signal in signals.values():
            for symbol, signal_value in strategy_signal.items():
                if symbol not in aggregated:
                    aggregated[symbol] = 0.0
                aggregated[symbol] += signal_value

        # Normalize and threshold
        for symbol in aggregated:
            aggregated[symbol] = (
                1.0
                if aggregated[symbol] > 0.5
                else (-1.0 if aggregated[symbol] < -0.5 else 0.0)
            )

        return aggregated

    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        try:
            ticker = await self.data_collector.get_ticker(symbol)
            return ticker.get("last") if ticker else None
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return None

    def _calculate_stop_loss(self, symbol: str, price: float, signal: float) -> float:
        """Calculate stop loss price."""
        if signal > 0:  # Long position
            return price * (1 - self.config.get("stop_loss_pct", 0.02))
        else:  # Short position
            return price * (1 + self.config.get("stop_loss_pct", 0.02))

    def _calculate_take_profit(self, entry_price: float, stop_loss: float) -> float:
        """Calculate take profit price."""
        risk = abs(entry_price - stop_loss)
        reward = risk * self.config.get("risk_reward_ratio", 2.0)

        if entry_price > stop_loss:  # Long position
            return entry_price + reward
        else:  # Short position
            return entry_price - reward

    async def _calculate_position_size(
        self, symbol: str, price: float, stop_loss: float, signal_strength: float
    ) -> float:
        """Calculate position size based on risk parameters."""
        try:
            # Get portfolio value
            portfolio_value = self.portfolio.get_total_value()

            # Get volatility (simplified)
            volatility = 0.2  # Should be calculated from historical data

            # Calculate position size using risk manager
            position_size = self.risk_manager.calculate_position_size(
                symbol=symbol,
                entry_price=price,
                stop_loss_price=stop_loss,
                portfolio_value=portfolio_value,
                volatility=volatility,
                correlation=0.0,  # Should be calculated
            )

            # Adjust for signal strength
            position_size *= abs(signal_strength)

            # Ensure minimum position size
            min_size = self.config.get("min_position_size", 0.0)
            if position_size < min_size:
                return 0.0

            return position_size

        except Exception as e:
            logger.error(f"Error calculating position size: {e}", exc_info=True)
            return 0.0

    async def _get_returns_data(
        self,
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Get historical returns and covariance matrix for optimization."""
        try:
            # In a real system, we'd get this from a data store
            symbols = self.config.get("symbols", ["BTC", "ETH", "SOL", "BNB", "XRP"])
            returns_data = {}

            # Get historical data for each symbol
            for symbol in symbols:
                ohlcv = await self.data_collector.get_ohlcv(
                    symbol=symbol, timeframe=Timeframe.DAILY, limit=90  # Last 90 days
                )

                if ohlcv and not ohlcv.empty:
                    # Calculate daily returns
                    returns = ohlcv["close"].pct_change().dropna()
                    returns_data[symbol] = returns

            if not returns_data:
                return None, None

            # Create DataFrame of returns
            returns_df = pd.DataFrame(returns_data)

            # Calculate covariance matrix
            cov_matrix = returns_df.cov() * 252  # Annualize

            return returns_df, cov_matrix

        except Exception as e:
            logger.error(f"Error getting returns data: {e}", exc_info=True)
            return None, None

    async def _rebalance_portfolio(self, target_weights: Dict[str, float]) -> None:
        """Rebalance portfolio to match target weights."""
        try:
            # Calculate current weights
            current_weights = {}
            total_value = self.portfolio.get_total_value()

            for symbol, position in self.portfolio.positions.items():
                position_value = position.quantity * position.current_price
                current_weights[symbol] = position_value / total_value

            # Calculate required trades
            trades = []
            cash = self.portfolio.cash

            for symbol, target_weight in target_weights.items():
                current_weight = current_weights.get(symbol, 0.0)

                # Skip if weight is already close to target
                if abs(target_weight - current_weight) < 0.01:  # 1% threshold
                    continue

                # Calculate target value and quantity
                target_value = total_value * target_weight
                current_value = total_value * current_weight

                # Get current price
                current_price = await self._get_current_price(symbol)
                if current_price is None or current_price <= 0:
                    continue

                # Calculate quantity to trade
                quantity = (target_value - current_value) / current_price

                if abs(quantity) < 1e-6:  # Skip very small quantities
                    continue

                # Add to trades
                side = "buy" if quantity > 0 else "sell"
                trades.append(
                    {
                        "symbol": symbol,
                        "side": side,
                        "quantity": abs(quantity),
                        "price": current_price,
                    }
                )

            # Execute trades
            for trade in trades:
                try:
                    await self.execution_handler.execute_order(trade)
                except Exception as e:
                    logger.error(f"Error executing rebalance trade: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"Error in portfolio rebalancing: {e}", exc_info=True)

    async def _handle_risk_limit_breach(self) -> None:
        """Take action when risk limits are breached."""
        try:
            # Close all positions
            for symbol, position in list(self.portfolio.positions.items()):
                if position.quantity == 0:
                    continue

                side = "sell" if position.quantity > 0 else "buy"
                order = {
                    "symbol": symbol,
                    "side": side,
                    "quantity": abs(position.quantity),
                    "price": position.current_price,
                    "reduce_only": True,
                }

                try:
                    await self.execution_handler.execute_order(order)
                except Exception as e:
                    logger.error(
                        f"Error closing position for {symbol}: {e}", exc_info=True
                    )

            logger.warning("All positions closed due to risk limit breach")

        except Exception as e:
            logger.error(f"Error handling risk limit breach: {e}", exc_info=True)

    async def run_backtest(
        self,
        data: pd.DataFrame,
        strategy_name: str = "moving_average",
        **backtest_params,
    ) -> BacktestResult:
        """Run a backtest on historical data."""
        try:
            # Get strategy
            strategy = self.strategies.get(strategy_name)
            if strategy is None:
                raise ValueError(f"Unknown strategy: {strategy_name}")

            # Configure backtest
            config = BacktestConfig(
                initial_capital=self.config.get("initial_capital", 100000.0),
                commission=self.config.get("commission", 0.0005),
                slippage=self.config.get("slippage", 0.0001),
                **backtest_params,
            )

            # Create backtest engine
            engine = AdvancedBacktestEngine(config)

            # Run backtest
            result = engine.run(
                data=data,
                generate_signals=strategy.generate_signals,
                strategy_params=strategy.get_params(),
            )

            return result

        except Exception as e:
            logger.error(f"Error running backtest: {e}", exc_info=True)
            raise

    async def shutdown(self) -> None:
        """Shut down the trading system gracefully."""
        if not self.running:
            return

        logger.info("Shutting down trading system...")
        self.running = False

        try:
            # Close all open positions
            await self._handle_risk_limit_breach()

            # Shutdown components
            if hasattr(self.data_collector, "close"):
                await self.data_collector.close()

            if hasattr(self.execution_handler, "close"):
                await self.execution_handler.close()

            logger.info("Trading system shutdown complete")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}", exc_info=True)

    def _handle_shutdown(self, signum, frame) -> None:
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        asyncio.create_task(self.shutdown())


def run() -> None:
    """Run the trading system."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("trading_system.log")],
    )

    # Suppress noisy loggers
    logging.getLogger("ccxt").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    # Configuration
    config = {
        "initial_capital": 100000.0,
        "symbols": ["BTC", "ETH", "SOL", "BNB", "XRP"],
        "stop_loss_pct": 0.02,  # 2% stop loss
        "take_profit_pct": 0.04,  # 4% take profit
        "risk_reward_ratio": 2.0,
        "commission": 0.0005,  # 5bps
        "slippage": 0.0001,  # 1bps
        "optimizer": "risk_parity",
        "min_position_size": 10.0,  # $10 minimum position size
    }

    # Create and run the trading system
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    trading_system = TradingSystem(config)

    try:
        loop.run_until_complete(trading_system.start())
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
    finally:
        # Cleanup
        pending = asyncio.all_tasks(loop=loop)
        for task in pending:
            task.cancel()

        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()

        logger.info("Application shutdown complete")


if __name__ == "__main__":
    run()
