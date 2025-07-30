"""
Advanced caching system for Tick Data Analysis & Alpha Detection.
"""

import json
from typing import Any, Optional, Dict, List
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from redis import Redis
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

class AdvancedCache:
    """Advanced caching system with Redis integration and LRU caching."""
    
    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0):
        """Initialize the cache system.
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
        """
        self.redis_client = Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=True
        )
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
        
    def cache_market_data(self, symbol: str, data: pd.DataFrame, 
                         expiry: int = 3600) -> None:
        """Cache market data with expiry.
        
        Args:
            symbol: Trading symbol
            data: Market data DataFrame
            expiry: Cache expiry in seconds
        """
        try:
            # Convert DataFrame to JSON
            data_json = data.to_json(orient='split')
            
            # Store in Redis with expiry
            self.redis_client.setex(
                f'market_data:{symbol}',
                expiry,
                data_json
            )
            
            # Update cache statistics
            self.cache_stats['hits'] += 1
            
        except Exception as e:
            logger.error(f"Error caching market data: {e}")
            raise
            
    def get_market_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Retrieve market data from cache.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            DataFrame if found, None otherwise
        """
        try:
            # Try to get from Redis
            data_json = self.redis_client.get(f'market_data:{symbol}')
            
            if data_json:
                # Convert JSON back to DataFrame
                data = pd.read_json(data_json, orient='split')
                self.cache_stats['hits'] += 1
                return data
                
            self.cache_stats['misses'] += 1
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving market data: {e}")
            return None
            
    def cache_technical_indicators(self, symbol: str, 
                                 indicators: Dict[str, pd.Series],
                                 expiry: int = 3600) -> None:
        """Cache technical indicators.
        
        Args:
            symbol: Trading symbol
            indicators: Dictionary of indicator names and Series
            expiry: Cache expiry in seconds
        """
        try:
            # Convert indicators to JSON
            indicators_json = {
                name: series.to_json() 
                for name, series in indicators.items()
            }
            
            # Store in Redis
            self.redis_client.setex(
                f'indicators:{symbol}',
                expiry,
                json.dumps(indicators_json)
            )
            
        except Exception as e:
            logger.error(f"Error caching indicators: {e}")
            raise
            
    def get_technical_indicators(self, symbol: str) -> Optional[Dict[str, pd.Series]]:
        """Retrieve technical indicators from cache.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary of indicators if found, None otherwise
        """
        try:
            # Try to get from Redis
            indicators_json = self.redis_client.get(f'indicators:{symbol}')
            
            if indicators_json:
                # Convert JSON back to indicators
                indicators = {
                    name: pd.read_json(series_json)
                    for name, series_json in json.loads(indicators_json).items()
                }
                return indicators
                
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving indicators: {e}")
            return None
            
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics.
        
        Returns:
            Dictionary of cache statistics
        """
        return self.cache_stats
        
    def clear_cache(self, pattern: str = '*') -> None:
        """Clear cache entries matching pattern.
        
        Args:
            pattern: Redis key pattern to match
        """
        try:
            # Find matching keys
            keys = self.redis_client.keys(pattern)
            
            if keys:
                # Delete matching keys
                self.redis_client.delete(*keys)
                self.cache_stats['evictions'] += len(keys)
                
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            raise
            
    def warm_cache(self, symbols: List[str]) -> None:
        """Warm cache with data for given symbols.
        
        Args:
            symbols: List of symbols to warm cache for
        """
        try:
            for symbol in symbols:
                # Get market data
                data = self.get_market_data(symbol)
                
                if data is not None:
                    # Cache technical indicators
                    indicators = {
                        'sma_20': data['close'].rolling(20).mean(),
                        'sma_50': data['close'].rolling(50).mean(),
                        'rsi': self._calculate_rsi(data['close'])
                    }
                    
                    self.cache_technical_indicators(symbol, indicators)
                    
        except Exception as e:
            logger.error(f"Error warming cache: {e}")
            raise
            
    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI technical indicator.
        
        Args:
            prices: Price series
            period: RSI period
            
        Returns:
            RSI series
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        return 100 - (100 / (1 + rs))
