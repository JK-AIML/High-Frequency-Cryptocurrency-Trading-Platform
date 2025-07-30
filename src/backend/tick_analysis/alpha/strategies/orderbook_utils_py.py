"""
Pure Python implementation of order book utilities.
This module will be used if the Cython version is not available.
"""
from typing import Dict, List, Tuple, Deque, DefaultDict, Optional, Any
from collections import defaultdict, deque
import numpy as np

class OrderBook:
    """In-memory order book implementation."""
    
    def __init__(self, max_depth: int = 10):
        self.bids: Dict[float, float] = {}  # price -> size
        self.asks: Dict[float, float] = {}  # price -> size
        self.max_depth = max_depth
        self._sorted_bids: List[float] = []
        self._sorted_asks: List[float] = []
        self._needs_sort = True
    
    def update(self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]) -> None:
        """Update the order book with new bid/ask data."""
        self.bids.clear()
        self.asks.clear()
        
        for price, size in bids:
            if size > 0:
                self.bids[price] = size
        
        for price, size in asks:
            if size > 0:
                self.asks[price] = size
        
        self._needs_sort = True
    
    def get_top_of_book(self) -> Tuple[float, float]:
        """Get best bid and ask prices and sizes."""
        self._ensure_sorted()
        best_bid = self._sorted_bids[0] if self._sorted_bids else 0
        best_ask = self._sorted_asks[0] if self._sorted_asks else float('inf')
        return (best_bid, self.bids.get(best_bid, 0)), (best_ask, self.asks.get(best_ask, 0))
    
    def get_depth(self) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """Get full depth of the order book up to max_depth."""
        self._ensure_sorted()
        bid_prices = self._sorted_bids[:self.max_depth]
        ask_prices = self._sorted_asks[:self.max_depth]
        
        bid_depth = [(p, self.bids[p]) for p in bid_prices]
        ask_depth = [(p, self.asks[p]) for p in ask_prices]
        
        return bid_depth, ask_depth
    
    def get_microstructure_features(self) -> Dict[str, float]:
        """Calculate market microstructure features."""
        bid_prices = np.array(sorted(self.bids.keys(), reverse=True))
        ask_prices = np.array(sorted(self.asks.keys()))
        
        if len(bid_prices) == 0 or len(ask_prices) == 0:
            return {}
        
        bid_sizes = np.array([self.bids[p] for p in bid_prices])
        ask_sizes = np.array([self.asks[p] for p in ask_prices])
        
        # Calculate order book imbalance
        total_bid = np.sum(bid_sizes)
        total_ask = np.sum(ask_sizes)
        ob_imbalance = (total_bid - total_ask) / (total_bid + total_ask + 1e-10)
        
        # Calculate spread
        spread = ask_prices[0] - bid_prices[0]
        mid_price = (bid_prices[0] + ask_prices[0]) / 2
        
        # Calculate order book slope
        def calc_slope(prices, sizes, ref_price, is_bid=True):
            if len(prices) < 2:
                return 0.0
            
            if is_bid:
                dist = ref_price - prices
            else:
                dist = prices - ref_price
                
            weights = sizes / np.sum(sizes)
            slope = np.sum(weights * dist) / (ref_price + 1e-10)
            return slope
        
        bid_slope = calc_slope(bid_prices, bid_sizes, bid_prices[0], True)
        ask_slope = calc_slope(ask_prices, ask_sizes, ask_prices[0], False)
        
        return {
            'spread': spread,
            'mid_price': mid_price,
            'ob_imbalance': ob_imbalance,
            'bid_slope': bid_slope,
            'ask_slope': ask_slope,
            'total_bid': total_bid,
            'total_ask': total_ask,
        }
    
    def _ensure_sorted(self) -> None:
        """Ensure the price levels are sorted."""
        if self._needs_sort:
            self._sorted_bids = sorted(self.bids.keys(), reverse=True)
            self._sorted_asks = sorted(self.asks.keys())
            self._needs_sort = False


def detect_latency_arbitrage(
    order_books: Dict[str, 'OrderBook'],
    symbols: List[str],
    latency_ms: float = 1.0
) -> List[Dict[str, Any]]:
    """
    Detect potential latency arbitrage opportunities across symbols.
    
    Args:
        order_books: Dictionary of symbol -> OrderBook
        symbols: List of symbol pairs to check (e.g., [('BTC', 'ETH')])
        latency_ms: Assumed latency in milliseconds
        
    Returns:
        List of potential arbitrage opportunities
    """
    opportunities = []
    
    for base, quote in symbols:
        if base not in order_books or quote not in order_books:
            continue
            
        base_book = order_books[base]
        quote_book = order_books[quote]
        
        # Get top of book
        (bid_px, bid_sz), (ask_px, ask_sz) = base_book.get_top_of_book()
        (quote_bid_px, _), (quote_ask_px, _) = quote_book.get_top_of_book()
        
        # Simple triangular arbitrage check
        # TODO: Implement more sophisticated cross-exchange checks
        if bid_px > 0 and ask_px > 0 and quote_bid_px > 0 and quote_ask_px > 0:
            # Calculate synthetic prices
            synthetic_bid = bid_px * quote_bid_px
            synthetic_ask = ask_px * quote_ask_px
            
            # Simple arbitrage check (can be enhanced with fees, slippage, etc.)
            if synthetic_bid > 1.001 * synthetic_ask:  # 0.1% threshold
                opportunities.append({
                    'type': 'triangular_arb',
                    'base': base,
                    'quote': quote,
                    'synthetic_bid': synthetic_bid,
                    'synthetic_ask': synthetic_ask,
                    'edge': (synthetic_bid - synthetic_ask) / synthetic_ask,
                    'latency_ms': latency_ms
                })
    
    return opportunities
