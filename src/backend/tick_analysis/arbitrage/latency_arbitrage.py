"""
Latency Arbitrage Detection Module

This module provides tools for detecting latency arbitrage opportunities
across different exchanges or trading venues. It includes features for
monitoring price discrepancies, calculating potential profits, and
evaluating execution feasibility.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import time
from concurrent.futures import ThreadPoolExecutor
import asyncio

logger = logging.getLogger(__name__)

@dataclass
class ArbitrageOpportunity:
    """Represents a detected arbitrage opportunity."""
    timestamp: datetime
    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_price: float
    sell_price: float
    spread_bps: float
    estimated_qty: float
    estimated_pnl: float
    min_profit_threshold: float = 0.0001  # Minimum profit threshold in quote currency
    max_slippage_bps: float = 5.0  # Maximum allowed slippage in basis points
    min_volume_usd: float = 1000.0  # Minimum notional volume for consideration
    
    def is_viable(self, fees_bps: float = 5.0) -> bool:
        """
        Check if the arbitrage opportunity is viable after considering fees.
        
        Args:
            fees_bps: Total round-trip fees in basis points
            
        Returns:
            bool: True if the opportunity is viable
        """
        # Convert bps to decimal
        fees_pct = fees_bps / 10000
        
        # Calculate net spread after fees
        net_spread = self.spread_bps / 10000 - fees_pct
        
        # Calculate net PnL
        net_pnl = self.estimated_pnl * (1 - fees_pct)
        
        # Check against thresholds
        return (net_spread > 0 and 
                net_pnl > self.min_profit_threshold and
                self.estimated_qty * self.buy_price > self.min_volume_usd)

class LatencyArbitrageDetector:
    """
    Detects latency arbitrage opportunities across multiple exchanges.
    
    This class monitors price feeds from different exchanges and identifies
    potential arbitrage opportunities based on price discrepancies that exceed
    trading costs and estimated execution risks.
    """
    
    def __init__(self, 
                 max_latency_ms: float = 100.0,
                 min_profit_threshold: float = 0.0001,
                 max_slippage_bps: float = 5.0,
                 min_volume_usd: float = 1000.0):
        """
        Initialize the LatencyArbitrageDetector.
        
        Args:
            max_latency_ms: Maximum allowed latency in milliseconds for an opportunity to be valid
            min_profit_threshold: Minimum profit threshold in quote currency
            max_slippage_bps: Maximum allowed slippage in basis points
            min_volume_usd: Minimum notional volume for consideration in USD
        """
        self.max_latency_ms = max_latency_ms
        self.min_profit_threshold = min_profit_threshold
        self.max_slippage_bps = max_slippage_bps
        self.min_volume_usd = min_volume_usd
        
        # Internal state
        self.last_prices: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.order_books: Dict[str, Dict[str, object]] = defaultdict(dict)
        self.last_update: Dict[str, datetime] = {}
        self.symbols: Set[str] = set()
        self.exchanges: Set[str] = set()
        
    def update_order_book(self, exchange: str, symbol: str, bids: List[Tuple[float, float]], 
                         asks: List[Tuple[float, float]], timestamp: Optional[datetime] = None):
        """
        Update the order book for a specific exchange and symbol.
        
        Args:
            exchange: Exchange identifier (e.g., 'binance', 'coinbase')
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            bids: List of (price, size) tuples for bids
            asks: List of (price, size) tuples for asks
            timestamp: Optional timestamp of the update
        """
        timestamp = timestamp or datetime.utcnow()
        
        # Update order book
        if symbol not in self.order_books[exchange]:
            from ..market_microstructure.order_book import OrderBook
            self.order_books[exchange][symbol] = OrderBook()
        
        self.order_books[exchange][symbol].update(bids, asks, timestamp=timestamp)
        
        # Update last price (use mid price)
        if bids and asks:
            bid_price = bids[0][0]
            ask_price = asks[0][0]
            mid_price = (bid_price + ask_price) / 2
            self.last_prices[exchange][symbol] = mid_price
            self.last_update[f"{exchange}:{symbol}"] = timestamp
            
        # Update tracking sets
        self.symbols.add(symbol)
        self.exchanges.add(exchange)
    
    def detect_opportunities(self, symbol: Optional[str] = None) -> List[ArbitrageOpportunity]:
        """
        Detect arbitrage opportunities across all exchanges and symbols.
        
        Args:
            symbol: Optional symbol to check (if None, checks all symbols)
            
        Returns:
            List of ArbitrageOpportunity objects
        """
        opportunities = []
        symbols = [symbol] if symbol else self.symbols
        
        for sym in symbols:
            # Get all exchanges that have this symbol
            exchanges = [e for e in self.exchanges if sym in self.last_prices.get(e, {})]
            
            # Skip if we don't have enough exchanges
            if len(exchanges) < 2:
                continue
                
            # Find best bid and ask across exchanges
            best_bid = (None, 0.0, 0.0)  # (exchange, price, qty)
            best_ask = (None, float('inf'), 0.0)  # (exchange, price, qty)
            
            for exchange in exchanges:
                if sym not in self.order_books.get(exchange, {}):
                    continue
                    
                ob = self.order_books[exchange][sym]
                
                # Get best bid
                if ob.bids:
                    bid_price, bid_level = next(iter(ob.bids.items()))
                    if bid_price > best_bid[1]:
                        best_bid = (exchange, bid_price, bid_level.size)
                
                # Get best ask
                if ob.asks:
                    ask_price, ask_level = next(iter(ob.asks.items()))
                    if ask_price < best_ask[1]:
                        best_ask = (exchange, ask_price, ask_level.size)
            
            # Check if we found valid quotes
            if best_bid[0] is None or best_ask[0] is None:
                continue
                
            # Skip if both quotes are from the same exchange
            if best_bid[0] == best_ask[0]:
                continue
                
            # Calculate spread in basis points
            spread = best_bid[1] - best_ask[1]  # Should be negative for arbitrage
            spread_bps = (spread / best_ask[1]) * 10000
            
            # Calculate estimated quantity (minimum of bid/ask quantities)
            qty = min(best_bid[2], best_ask[2])
            
            # Calculate estimated P&L (negative for arbitrage)
            pnl = spread * qty
            
            # Create opportunity
            opp = ArbitrageOpportunity(
                timestamp=datetime.utcnow(),
                symbol=sym,
                buy_exchange=best_ask[0],
                sell_exchange=best_bid[0],
                buy_price=best_ask[1],
                sell_price=best_bid[1],
                spread_bps=abs(spread_bps),
                estimated_qty=qty,
                estimated_pnl=abs(pnl),
                min_profit_threshold=self.min_profit_threshold,
                max_slippage_bps=self.max_slippage_bps,
                min_volume_usd=self.min_volume_usd
            )
            
            # Check if opportunity is viable
            if opp.is_viable():
                opportunities.append(opp)
        
        return opportunities
    
    async def monitor_opportunities(self, 
                                   callback,
                                   symbols: Optional[List[str]] = None,
                                   interval_sec: float = 1.0):
        """
        Continuously monitor for arbitrage opportunities.
        
        Args:
            callback: Function to call when an opportunity is found
            symbols: List of symbols to monitor (None for all)
            interval_sec: Monitoring interval in seconds
        """
        while True:
            try:
                opportunities = self.detect_opportunities()
                for opp in opportunities:
                    if symbols is None or opp.symbol in symbols:
                        await callback(opp)
                
                await asyncio.sleep(interval_sec)
                
            except Exception as e:
                logger.error(f"Error in monitor_opportunities: {e}", exc_info=True)
                await asyncio.sleep(1)  # Prevent tight loop on errors
