"""
Order Book Reconstruction Module

This module provides tools for reconstructing and analyzing limit order book (LOB) data.
It includes features for tracking price levels, calculating depth profiles, and detecting
liquidity imbalances.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Deque, DefaultDict
import numpy as np
import pandas as pd
from collections import deque, defaultdict, OrderedDict
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

@dataclass
class OrderBookLevel:
    """Represents a single price level in the order book."""
    price: float
    size: float = 0.0
    count: int = 0
    last_updated: Optional[datetime] = None

class OrderBook:
    """
    Reconstructs and maintains a limit order book from incremental updates.
    
    This class provides a complete implementation of an order book that can be
    updated with market data events and provides various analytics on the book state.
    """
    
    def __init__(self, max_levels: int = 10, tick_size: float = 0.01):
        """
        Initialize the OrderBook.
        
        Args:
            max_levels: Maximum number of price levels to maintain on each side
            tick_size: Minimum price increment for the instrument
        """
        self.max_levels = max_levels
        self.tick_size = tick_size
        self.reset()
    
    def reset(self) -> None:
        """Reset the order book to empty state."""
        self.bids: Dict[float, OrderBookLevel] = OrderedDict()
        self.asks: Dict[float, OrderBookLevel] = OrderedDict()
        self.trades: Deque[Tuple[float, float, bool, datetime]] = deque(maxlen=10000)
        self.last_update: Optional[datetime] = None
        self.last_sequence: int = 0
    
    def update(self, bids: List[Tuple[float, float]], 
               asks: List[Tuple[float, float]],
               sequence: Optional[int] = None,
               timestamp: Optional[datetime] = None) -> None:
        """
        Update the order book with new price levels.
        
        Args:
            bids: List of (price, size) tuples for bid updates
            asks: List of (price, size) tuples for ask updates
            sequence: Optional sequence number for updates
            timestamp: Optional timestamp of the update
        """
        if sequence is not None and sequence <= self.last_sequence:
            return  # Ignore old updates
            
        self.last_sequence = sequence or (self.last_sequence + 1)
        timestamp = timestamp or datetime.utcnow()
        
        # Update bids
        for price, size in bids:
            price = round(price / self.tick_size) * self.tick_size  # Align to tick size
            if np.isclose(size, 0):
                self.bids.pop(price, None)
            else:
                if price not in self.bids:
                    self.bids[price] = OrderBookLevel(price=price, size=0, count=0)
                self.bids[price].size = size
                self.bids[price].count += 1
                self.bids[price].last_updated = timestamp
        
        # Update asks
        for price, size in asks:
            price = round(price / self.tick_size) * self.tick_size  # Align to tick size
            if np.isclose(size, 0):
                self.asks.pop(price, None)
            else:
                if price not in self.asks:
                    self.asks[price] = OrderBookLevel(price=price, size=0, count=0)
                self.asks[price].size = size
                self.asks[price].count += 1
                self.asks[price].last_updated = timestamp
        
        # Sort bids in descending order and asks in ascending order
        self.bids = OrderedDict(sorted(self.bids.items(), reverse=True))
        self.asks = OrderedDict(sorted(self.asks.items()))
        
        # Trim to max levels
        self.bids = OrderedDict(list(self.bids.items())[:self.max_levels])
        self.asks = OrderedDict(list(self.asks.items())[:self.max_levels])
        
        self.last_update = timestamp
    
    def add_trade(self, price: float, size: float, is_buy: bool, 
                 timestamp: Optional[datetime] = None) -> None:
        """
        Add a trade to the order book history.
        
        Args:
            price: Trade price
            size: Trade size
            is_buy: Whether the trade was buyer-initiated
            timestamp: Optional timestamp of the trade
        """
        timestamp = timestamp or datetime.utcnow()
        self.trades.append((price, size, is_buy, timestamp))
    
    def get_mid_price(self) -> Optional[float]:
        """Get the current mid price (average of best bid and ask)."""
        if not self.bids or not self.asks:
            return None
        best_bid = next(iter(self.bids.keys()))
        best_ask = next(iter(self.asks.keys()))
        return (best_bid + best_ask) / 2
    
    def get_spread(self) -> Optional[float]:
        """Get the current bid-ask spread."""
        if not self.bids or not self.asks:
            return None
        best_bid = next(iter(self.bids.keys()))
        best_ask = next(iter(self.asks.keys()))
        return best_ask - best_bid
    
    def get_weighted_mid_price(self) -> Optional[float]:
        """Get the volume-weighted mid price."""
        if not self.bids or not self.asks:
            return None
            
        best_bid_price, best_bid = next(iter(self.bids.items()))
        best_ask_price, best_ask = next(iter(self.asks.items()))
        
        total_size = best_bid.size + best_ask.size
        if total_size == 0:
            return (best_bid_price + best_ask_price) / 2
            
        return (best_bid_price * best_ask.size + best_ask_price * best_bid.size) / total_size
    
    def get_imbalance(self, levels: int = 5) -> float:
        """
        Calculate the order book imbalance.
        
        Args:
            levels: Number of price levels to consider
            
        Returns:
            float: Imbalance ratio between -1 (bearish) and 1 (bullish)
        """
        bid_vol = sum(level.size for level in list(self.bids.values())[:levels])
        ask_vol = sum(level.size for level in list(self.asks.values())[:levels])
        
        if bid_vol + ask_vol == 0:
            return 0.0
            
        return (bid_vol - ask_vol) / (bid_vol + ask_vol)
    
    def get_liquidity_horizon(self, target_size: float, side: str) -> Optional[float]:
        """
        Calculate the price impact of executing a target size.
        
        Args:
            target_size: Size to execute
            side: 'buy' or 'sell'
            
        Returns:
            float: Volume-weighted average price impact in ticks
        """
        if side.lower() not in ['buy', 'sell']:
            raise ValueError("Side must be either 'buy' or 'sell'")
            
        levels = self.asks if side.lower() == 'buy' else self.bids
        if not levels:
            return None
            
        remaining_size = target_size
        total_value = 0.0
        
        for price, level in levels.items():
            if remaining_size <= 0:
                break
                
            size_to_take = min(remaining_size, level.size)
            total_value += price * size_to_take
            remaining_size -= size_to_take
        
        if target_size == 0:
            return 0.0
            
        vwap = total_value / (target_size - remaining_size)
        best_price = next(iter(levels.keys()))
        
        if side.lower() == 'buy':
            return (vwap - best_price) / self.tick_size
        else:
            return (best_price - vwap) / self.tick_size
    
    def get_order_flow_imbalance(self, window: int = 100) -> float:
        """
        Calculate the order flow imbalance over a window of trades.
        
        Args:
            window: Number of trades to consider
            
        Returns:
            float: Order flow imbalance ratio (-1 to 1)
        """
        if not self.trades or window == 0:
            return 0.0
            
        recent_trades = list(self.trades)[-window:]
        buy_vol = sum(size for _, size, is_buy, _ in recent_trades if is_buy)
        sell_vol = sum(size for _, size, is_buy, _ in recent_trades if not is_buy)
        
        if buy_vol + sell_vol == 0:
            return 0.0
            
        return (buy_vol - sell_vol) / (buy_vol + sell_vol)
    
    def snapshot(self) -> dict:
        """
        Create a snapshot of the current order book state.
        
        Returns:
            dict: Dictionary containing order book state
        """
        return {
            'timestamp': self.last_update or datetime.utcnow(),
            'sequence': self.last_sequence,
            'bids': [(price, level.size) for price, level in self.bids.items()],
            'asks': [(price, level.size) for price, level in self.asks.items()],
            'mid_price': self.get_mid_price(),
            'spread': self.get_spread(),
            'weighted_mid': self.get_weighted_mid_price(),
            'imbalance': self.get_imbalance(),
            'buy_liq_horizon': self.get_liquidity_horizon(10, 'buy'),
            'sell_liq_horizon': self.get_liquidity_horizon(10, 'sell'),
            'order_flow_imbalance': self.get_order_flow_imbalance()
        }
