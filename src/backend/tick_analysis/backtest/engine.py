"""Backtest engine for tick analysis, supporting both backtesting (using data providers) and live trading (using exchange adapters)."""

import vectorbt as vbt
import pandas as pd
from tick_analysis.execution.order import Order, OrderSide, OrderType, TimeInForce
from tick_analysis.portfolio import Portfolio
from tick_analysis.monitoring.monitoring import MetricsCollector, AlertSeverity, AlertType

class BacktestEngine:
    def __init__(self, initial_capital=None, data_provider=None, commission=None, slippage=None, benchmark=None, exchange=None, *args, **kwargs):
        """
        Args:
            initial_capital (float): Starting capital for backtest/live trading.
            data_provider (ExchangeInterface): Adapter for historical/market data (CryptoCompare, Polygon, etc.).
            commission (float): Commission per trade.
            slippage (float): Slippage per trade.
            benchmark: Optional benchmark for performance comparison.
            exchange (ExchangeInterface): Adapter for live trading (Binance, etc.).
        """
        self.initial_capital = float(initial_capital) if initial_capital is not None else 100_000.0
        self.data_provider = data_provider
        self.commission = float(commission) if commission is not None else 0.001
        self.slippage = float(slippage) if slippage is not None else 0.0
        self.benchmark = benchmark
        self.exchange = exchange  # Only used for live trading
        self.portfolio = None  # Will be set after running backtest

    def run(self, strategy=None, symbols=None, start_date=None, end_date=None, timeframe="1h", data: pd.DataFrame = None, live=False, **kwargs):
        """
        Run a backtest or live trading session.
        Args:
            strategy: Callable for generating signals.
            symbols: List of symbols.
            start_date, end_date: Date range for data.
            timeframe: Data frequency.
            data: Optional preloaded DataFrame.
            live (bool): If True, use exchange for live trading. If False, run backtest.
        Returns:
            dict: Results of the run.
        """
        # Load data
        if data is None:
            if self.data_provider is not None and symbols is not None:
                data = self.data_provider.get_historical_data(symbols[0], start_date, end_date, timeframe)
            else:
                raise ValueError("Must provide either data or data_provider and symbols")
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")
        close = data["close"] if "close" in data else data.iloc[:, 0]
        # Generate signals
        if strategy is not None:
            entries, exits = self._generate_signals(strategy, data)
        else:
            # Default: buy and hold
            entries = pd.Series([True] + [False]*(len(close)-1), index=close.index)
            exits = pd.Series([False]*(len(close)-1) + [True], index=close.index)
        # Backtest or live trading
        if not live:
            pf = vbt.Portfolio.from_signals(
                close,
                entries,
                exits,
                init_cash=self.initial_capital,
                fees=self.commission,
                slippage=self.slippage,
                freq=timeframe if isinstance(timeframe, str) else None
            )
            self.portfolio = pf
            results = {
                "returns": pf.returns().values,
                "positions": pf.positions.values if hasattr(pf, 'positions') else [],
                "trades": pf.trades.records_readable if hasattr(pf, 'trades') else [],
                "benchmark_returns": close.pct_change().fillna(0).values,
                "portfolio_values": pf.value().values,
            }
        else:
            if self.exchange is None:
                raise ValueError("Exchange adapter must be provided for live trading.")
            results = {"orders": []}
            for i, (entry, exit) in enumerate(zip(entries, exits)):
                ts = close.index[i]
                price = close.iloc[i]
                if entry:
                    order = self.exchange.place_order(symbols[0], "BUY", 1, "MARKET", price=None)
                    results["orders"].append({"timestamp": ts, "action": "BUY", "order": order})
                if exit:
                    order = self.exchange.place_order(symbols[0], "SELL", 1, "MARKET", price=None)
                    results["orders"].append({"timestamp": ts, "action": "SELL", "order": order})
        return results

    def rebalance_portfolio(self, target_weights: dict, prices: dict, min_trade: float = 1e-6, metrics: MetricsCollector = None, alert_threshold: float = 0.2):
        """
        Live portfolio rebalancing: generate and execute orders to reach target weights.
        Integrates with monitoring and alerting.
        Args:
            target_weights: dict of {symbol: target_weight (0-1)}
            prices: dict of {symbol: current_price}
            min_trade: minimum trade value to avoid dust
            metrics: MetricsCollector instance for Prometheus metrics
            alert_threshold: Fractional deviation from target to trigger alert
        Returns:
            list of executed order results
        Usage:
            engine.rebalance_portfolio({'BTC': 0.5, 'ETH': 0.3, 'SOL': 0.2}, {'BTC': 30000, 'ETH': 2000, 'SOL': 100}, metrics=metrics)
        """
        if self.portfolio is None:
            self.portfolio = Portfolio(initial_cash=self.initial_capital)
        orders = self.portfolio.rebalance_to_target(target_weights, prices, min_trade=min_trade)
        results = []
        # Calculate current weights for alerting
        self.portfolio.update_equity()
        total_value = float(self.portfolio.equity)
        current_values = {sym: float(pos.quantity) * prices.get(sym, 0) for sym, pos in self.portfolio.positions.items()}
        current_weights = {sym: (current_values.get(sym, 0) / total_value if total_value > 0 else 0) for sym in target_weights}
        for order in orders:
            if self.exchange is not None:
                result = self.exchange.place_order(order.symbol, order.side.name, float(order.quantity), order.order_type.name, price=float(order.price))
                results.append(result)
                self.portfolio.execute_order(order)
                # Monitoring: record trade size
                if metrics is not None:
                    metrics.record_metric('trade_volume', float(order.quantity) * float(order.price), labels={'symbol': order.symbol})
                # Alerting: trigger alert if trade is large or deviation is high
                deviation = abs(current_weights.get(order.symbol, 0) - target_weights.get(order.symbol, 0))
                if deviation > alert_threshold and metrics is not None:
                    metrics.trigger_alert(
                        alert_type=AlertType.BUSINESS,
                        severity=AlertSeverity.WARNING,
                        message=f"Large rebalance trade for {order.symbol}: deviation {deviation:.2%}",
                        symbol=order.symbol,
                        deviation=deviation,
                        quantity=float(order.quantity),
                        price=float(order.price)
                    )
        return results

    def calculate_risk_metrics(self, returns=None, **kwargs):
        if returns is None and self.portfolio is not None:
            returns = self.portfolio.returns().values
        elif returns is None:
            return {}
        import numpy as np
        total_return = np.prod(1 + returns) - 1 if len(returns) > 0 else 0.0
        annual_return = self._annualized_return(returns)
        volatility = np.std(returns) * np.sqrt(252)
        sharpe_ratio = (annual_return / volatility) if volatility != 0 else 0.0
        max_drawdown = self._max_drawdown(returns)
        calmar_ratio = (annual_return / abs(max_drawdown)) if max_drawdown != 0 else 0.0
        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar_ratio
        }

    def _generate_signals(self, strategy, data):
        # The strategy function should return entries and exits as boolean Series
        result = strategy(self, data)
        if isinstance(result, tuple) and len(result) == 2:
            entries, exits = result
        else:
            # Fallback: treat as entries only, exit at end
            entries = pd.Series([True] + [False]*(len(data)-1), index=data.index)
            exits = pd.Series([False]*(len(data)-1) + [True], index=data.index)
        return entries, exits

    def _annualized_return(self, returns):
        import numpy as np
        n = len(returns)
        if n == 0:
            return 0.0
        return (np.prod(1 + returns) ** (252 / n)) - 1

    def _max_drawdown(self, returns):
        import numpy as np
        cumulative = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        return drawdown.min() if len(drawdown) > 0 else 0.0

    def _get_benchmark_data(self, *args, **kwargs):
        return []

    def _execute_order(self, order, current_price, timestamp):
        from tick_analysis.execution.order import Trade
        commission = order.quantity * current_price * self.commission
        return Trade(
            symbol=order.symbol,
            order_type=order.order_type,
            side=order.side,
            quantity=order.quantity,
            price=current_price,
            timestamp=timestamp,
            order_id=order.id,
            commission=commission,
        )
