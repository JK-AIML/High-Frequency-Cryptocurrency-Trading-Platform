"""
Time-series database storage module for tick data analysis.

This module provides interfaces and implementations for storing and querying
tick data in a time-series optimized database (TimescaleDB).
"""

from .tsdb import TimeSeriesDB, TimeSeriesQuery, TimeRange, AggregationType
from .models import (
    TickDataPoint,
    OHLCV,
    OrderBookSnapshot,
    Trade,
    MarketDepth
)

__all__ = [
    'TimeSeriesDB',
    'TimeSeriesQuery',
    'TimeRange',
    'AggregationType',
    'TickDataPoint',
    'OHLCV',
    'OrderBookSnapshot',
    'Trade',
    'MarketDepth'
]
