"""
Order Flow Analysis Module

This module provides tools for analyzing order flow dynamics in financial markets.
It includes features for tracking order book imbalances, order flow toxicity, and
market participant behavior.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from collections import deque, defaultdict
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class OrderFlowAnalyzer:
    """
    Analyzes order flow dynamics including order book imbalances and flow toxicity.
    
    This class tracks various order flow metrics that can be used to detect
    market maker behavior, large participant activity, and potential price
    movements based on order flow dynamics.
    
    Attributes:
        window_size: Size of the rolling window for calculating flow metrics
        imbalance_window: Window size for order book imbalance calculations
        trade_imbalance_window: Window size for trade imbalance calculations
    """
    window_size: int = 1000
    imbalance_window: int = 20
    trade_imbalance_window: int = 100
    
    def __post_init__(self):
        """Initialize internal state variables."""
        self.reset()
    
    def reset(self):
        """Reset the analyzer's internal state."""
        # Order book state
        self.bid_volumes = deque(maxlen=self.window_size)
        self.ask_volumes = deque(maxlen=self.window_size)
        self.bid_counts = deque(maxlen=self.window_size)
        self.ask_counts = deque(maxlen=self.window_size)
        
        # Trade flow state
        self.buy_volumes = deque(maxlen=self.trade_imbalance_window)
        self.sell_volumes = deque(maxlen=self.trade_imbalance_window)
        
        # Time-based state
        self.last_update = None
        self.volume_profile = defaultdict(float)
    
    def update_order_book(self, bids: List[Tuple[float, float]], 
                         asks: List[Tuple[float, float]], 
                         timestamp: Optional[datetime] = None):
        """
        Update order book state with new bid/ask data.
        
        Args:
            bids: List of (price, size) tuples for bids
            asks: List of (price, size) tuples for asks
            timestamp: Optional timestamp of the update
        """
        if not bids or not asks:
            return
            
        bid_vol = sum(size for _, size in bids[:5])  # Top 5 levels
        ask_vol = sum(size for _, size in asks[:5])
        bid_count = len(bids)
        ask_count = len(asks)
        
        self.bid_volumes.append(bid_vol)
        self.ask_volumes.append(ask_vol)
        self.bid_counts.append(bid_count)
        self.ask_counts.append(ask_count)
        
        if timestamp:
            self.last_update = timestamp
    
    def update_trades(self, trades: List[Tuple[float, float, bool]], 
                      timestamp: Optional[datetime] = None):
        """
        Update trade flow state with new trade data.
        
        Args:
            trades: List of (price, size, is_buy) tuples
            timestamp: Optional timestamp of the update
        """
        if not trades:
            return
            
        for price, size, is_buy in trades:
            if is_buy:
                self.buy_volumes.append(size)
            else:
                self.sell_volumes.append(size)
            
            # Update volume profile
            if timestamp:
                price_level = round(price, 2)  # Adjust precision as needed
                self.volume_profile[price_level] += size
    
    def get_order_book_imbalance(self, window: Optional[int] = None) -> float:
        """
        Calculate the current order book imbalance.
        
        Args:
            window: Number of levels to consider (default: all available)
            
        Returns:
            float: Order book imbalance ratio (-1 to 1)
        """
        if not self.bid_volumes or not self.ask_volumes:
            return 0.0
            
        bid_vol = sum(list(self.bid_volumes)[-window or len(self.bid_volumes):])
        ask_vol = sum(list(self.ask_volumes)[-window or len(self.ask_volumes):])
        
        if bid_vol + ask_vol == 0:
            return 0.0
            
        return (bid_vol - ask_vol) / (bid_vol + ask_vol)
    
    def get_trade_imbalance(self, window: Optional[int] = None) -> float:
        """
        Calculate the trade flow imbalance.
        
        Args:
            window: Number of trades to consider (default: all available)
            
        Returns:
            float: Trade imbalance ratio (-1 to 1)
        """
        if not self.buy_volumes or not self.sell_volumes:
            return 0.0
            
        buy_vol = sum(list(self.buy_volumes)[-window or len(self.buy_volumes):])
        sell_vol = sum(list(self.sell_volumes)[-window or len(self.sell_volumes):])
        
        if buy_vol + sell_vol == 0:
            return 0.0
            
        return (buy_vol - sell_vol) / (buy_vol + sell_vol)
    
    def get_flow_toxicity(self) -> float:
        """
        Calculate order flow toxicity based on VPIN.
        
        Returns:
            float: VPIN (Volume-Synchronized Probability of Informed Trading)
        """
        if not self.buy_volumes or not self.sell_volumes:
            return 0.0
            
        # Simple VPIN implementation
        buy_vol = sum(self.buy_volumes)
        sell_vol = sum(self.sell_volumes)
        total_vol = buy_vol + sell_vol
        
        if total_vol == 0:
            return 0.0
            
        return abs(buy_vol - sell_vol) / total_vol
    
    def get_volume_profile(self, price_precision: int = 2) -> Dict[float, float]:
        """
        Get the current volume profile.
        
        Args:
            price_precision: Number of decimal places to round prices
            
        Returns:
            dict: Price level to volume mapping
        """
        if not self.volume_profile:
            return {}
            
        # Round prices to specified precision
        rounded_profile = defaultdict(float)
        for price, vol in self.volume_profile.items():
            rounded_price = round(price, price_precision)
            rounded_profile[rounded_price] += vol
            
        return dict(rounded_profile)
    
    def get_all_metrics(self) -> dict:
        """
        Get all available order flow metrics.
        
        Returns:
            dict: Dictionary containing all metrics
        """
        return {
            'order_book_imbalance': self.get_order_book_imbalance(),
            'short_term_imbalance': self.get_order_book_imbalance(window=10),
            'trade_imbalance': self.get_trade_imbalance(),
            'flow_toxicity': self.get_flow_toxicity(),
            'bid_volume': sum(self.bid_volumes) if self.bid_volumes else 0,
            'ask_volume': sum(self.ask_volumes) if self.ask_volumes else 0,
            'bid_count': len(self.bid_counts) if self.bid_counts else 0,
            'ask_count': len(self.ask_counts) if self.ask_counts else 0,
            'buy_volume': sum(self.buy_volumes) if self.buy_volumes else 0,
            'sell_volume': sum(self.sell_volumes) if self.sell_volumes else 0,
            'last_update': self.last_update or datetime.utcnow()
        }
