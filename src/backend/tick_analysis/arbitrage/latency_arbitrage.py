"""
Latency Arbitrage Detection Module

This module provides tools for detecting latency arbitrage opportunities
across different exchanges or trading venues. It includes features for
monitoring price discrepancies, calculating potential profits, and
evaluating execution feasibility.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
from datetime import datetime, timedelta
import logging
import time
import asyncio

# --- FIX 1: Imports moved to the top of the file for proper structure. ---
# Assuming a project structure where this relative import is valid.
# from ..market_microstructure.order_book import OrderBook
# For standalone execution, we can use a placeholder class.
class OrderBook:
    """Placeholder for a real OrderBook implementation."""
    def __init__(self):
        self.bids = {}
        self.asks = {}
    def update(self, bids, asks, timestamp):
        self.bids = {p: type('level', (), {'size': s}) for p, s in sorted(bids, key=lambda x: x[0], reverse=True)}
        self.asks = {p: type('level', (), {'size': s}) for p, s in sorted(asks, key=lambda x: x[0])}
    @property
    def best_bid(self):
        if not self.bids: return None, None
        price = next(iter(self.bids))
        return price, self.bids[price].size
    @property
    def best_ask(self):
        if not self.asks: return None, None
        price = next(iter(self.asks))
        return price, self.asks[price].size


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
        This is a simplified check. A production system would calculate slippage
        and more precise fee structures.
        
        Args:
            fees_bps: Total round-trip fees in basis points.
            
        Returns:
            bool: True if the opportunity is viable.
        """
        fees_decimal = fees_bps / 10000
        
        # Net spread in decimal form after fees
        net_spread_decimal = (self.spread_bps / 10000) - fees_decimal

        # PnL after applying fees to the total notional value
        # A more precise calculation would apply fees per leg.
        net_pnl = self.estimated_pnl * (1 - fees_decimal)

        # TODO: Implement a more advanced check here that simulates execution
        # across the order book to calculate expected slippage and ensures
        # it's below `self.max_slippage_bps`.

        return (net_spread_decimal > 0 and
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
                 max_latency_ms: float = 50.0,
                 min_profit_threshold: float = 0.0001,
                 max_slippage_bps: float = 5.0,
                 min_volume_usd: float = 1000.0):
        """
        Initialize the LatencyArbitrageDetector.
        
        Args:
            max_latency_ms: Maximum allowed latency in milliseconds between quotes for an opportunity to be valid.
            min_profit_threshold: Minimum profit threshold in quote currency.
            max_slippage_bps: Maximum allowed slippage in basis points.
            min_volume_usd: Minimum notional volume for consideration in USD.
        """
        self.max_latency_ms = max_latency_ms
        self.min_profit_threshold = min_profit_threshold
        self.max_slippage_bps = max_slippage_bps
        self.min_volume_usd = min_volume_usd

        # Internal state
        self.order_books: Dict[str, Dict[str, OrderBook]] = defaultdict(lambda: defaultdict(OrderBook))
        self.last_update: Dict[str, datetime] = {}
        self.symbols: Set[str] = set()
        self.exchanges: Set[str] = set()
        
        # --- FIX 2: Added a lock for concurrency safety. ---
        self.lock = asyncio.Lock()

    async def update_order_book(self, exchange: str, symbol: str, bids: List[Tuple[float, float]],
                                asks: List[Tuple[float, float]], timestamp: Optional[datetime] = None):
        """
        Update the order book for a specific exchange and symbol. This method is now async
        to handle the lock.
        
        Args:
            exchange: Exchange identifier (e.g., 'binance', 'coinbase').
            symbol: Trading pair symbol (e.g., 'BTC/USDT').
            bids: List of (price, size) tuples for bids.
            asks: List of (price, size) tuples for asks.
            timestamp: Optional timestamp of the update, ideally from the exchange.
        """
        async with self.lock:
            ts = timestamp or datetime.utcnow()
            
            self.order_books[exchange][symbol].update(bids, asks, timestamp=ts)
            self.last_update[f"{exchange}:{symbol}"] = ts

            self.symbols.add(symbol)
            self.exchanges.add(exchange)

    async def detect_opportunities(self, symbol: Optional[str] = None) -> List[ArbitrageOpportunity]:
        """
        Detect arbitrage opportunities across all exchanges and symbols. This method is now
        async to handle the lock.
        
        Args:
            symbol: Optional symbol to check (if None, checks all symbols).
            
        Returns:
            List of ArbitrageOpportunity objects.
        """
        opportunities = []
        symbols_to_check = [symbol] if symbol else list(self.symbols)

        async with self.lock:
            for sym in symbols_to_check:
                exchanges_with_sym = [e for e in self.exchanges if sym in self.order_books.get(e, {})]

                if len(exchanges_with_sym) < 2:
                    continue

                best_bid_info = {'price': 0.0, 'qty': 0.0, 'exchange': None}
                best_ask_info = {'price': float('inf'), 'qty': 0.0, 'exchange': None}

                for ex in exchanges_with_sym:
                    ob = self.order_books[ex][sym]
                    bid_price, bid_qty = ob.best_bid
                    ask_price, ask_qty = ob.best_ask

                    if bid_price is not None and bid_price > best_bid_info['price']:
                        best_bid_info.update(price=bid_price, qty=bid_qty, exchange=ex)
                    
                    if ask_price is not None and ask_price < best_ask_info['price']:
                        best_ask_info.update(price=ask_price, qty=ask_qty, exchange=ex)

                # --- FIX 3: Clearer arbitrage logic and added latency check. ---
                buy_exchange = best_ask_info['exchange']
                sell_exchange = best_bid_info['exchange']
                
                # Check for a valid opportunity (must be on different exchanges)
                if buy_exchange is None or sell_exchange is None or buy_exchange == sell_exchange:
                    continue

                # The core arbitrage condition: can we sell higher than we buy?
                if best_bid_info['price'] > best_ask_info['price']:
                    buy_price = best_ask_info['price']
                    sell_price = best_bid_info['price']

                    # Check latency between the two quotes
                    buy_ts = self.last_update.get(f"{buy_exchange}:{sym}")
                    sell_ts = self.last_update.get(f"{sell_exchange}:{sym}")

                    if buy_ts and sell_ts:
                        latency_diff_ms = abs((buy_ts - sell_ts).total_seconds() * 1000)
                        if latency_diff_ms > self.max_latency_ms:
                            # Stale quote, skip this opportunity
                            continue

                    # Calculate spread and potential profit
                    spread = sell_price - buy_price
                    spread_bps = (spread / buy_price) * 10000
                    
                    estimated_qty = min(best_ask_info['qty'], best_bid_info['qty'])
                    estimated_pnl = spread * estimated_qty

                    opp = ArbitrageOpportunity(
                        timestamp=datetime.utcnow(),
                        symbol=sym,
                        buy_exchange=buy_exchange,
                        sell_exchange=sell_exchange,
                        buy_price=buy_price,
                        sell_price=sell_price,
                        spread_bps=spread_bps,
                        estimated_qty=estimated_qty,
                        estimated_pnl=estimated_pnl,
                        min_profit_threshold=self.min_profit_threshold,
                        max_slippage_bps=self.max_slippage_bps,
                        min_volume_usd=self.min_volume_usd
                    )
                    
                    # Check if opportunity is viable after fees and other constraints
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
            callback: Async function to call when an opportunity is found.
            symbols: List of symbols to monitor (None for all).
            interval_sec: Monitoring interval in seconds.
        """
        while True:
            try:
                opportunities = await self.detect_opportunities()
                for opp in opportunities:
                    if symbols is None or opp.symbol in symbols:
                        await callback(opp)
                await asyncio.sleep(interval_sec)
            except Exception as e:
                logger.error(f"Error in monitor_opportunities: {e}", exc_info=True)
                await asyncio.sleep(5)  # Prevent tight loop on errors
