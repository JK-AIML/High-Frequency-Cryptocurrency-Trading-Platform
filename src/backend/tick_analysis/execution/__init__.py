"""Execution package for order handling and trade execution."""

# execution package
import tick_analysis.execution
from .order import Order, OrderSide, OrderType, TimeInForce, OrderStatus, PositionSide, Trade
from .handler import ExecutionHandler
from .exceptions import (
    ExecutionError,
    OrderRejected,
    InsufficientFunds,
    InvalidOrder,
    MarketClosed,
    PositionLimitExceeded
)
from .cryptocompare_adapter import CryptoCompareAdapter  # Market data only
from .binance_adapter import BinanceAdapter  # Market data + trading
from .polygon_adapter import PolygonAdapter  # Market data only

"""
Adapters:
- CryptoCompareAdapter: Market data provider only (no trading)
- PolygonAdapter: Market data provider only (no trading)
- BinanceAdapter: Market data + trading (use for live trading)
"""

__all__ = [
    "Order",
    "OrderSide",
    "OrderType",
    "TimeInForce",
    "OrderStatus",
    "PositionSide",
    "Trade",
    "ExecutionHandler",
    "ExecutionError",
    "OrderRejected",
    "InsufficientFunds",
    "InvalidOrder",
    "MarketClosed",
    "PositionLimitExceeded"
]
