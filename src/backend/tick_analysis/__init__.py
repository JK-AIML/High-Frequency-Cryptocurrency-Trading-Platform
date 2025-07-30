"""
Tick Data Analysis & Alpha Detection System

A comprehensive system for analyzing tick data and detecting alpha signals.
"""

__version__ = "0.1.0"

# Import key components for easier access
from tick_analysis.data.collector import DataCollector
from tick_analysis.strategies.base import BaseStrategy
from tick_analysis.execution.handler import ExecutionHandler
from tick_analysis.backtest.engine import BacktestEngine
from tick_analysis.portfolio import Portfolio

__all__ = [
    "DataCollector",
    "BaseStrategy",
    "ExecutionHandler",
    "BacktestEngine",
]
