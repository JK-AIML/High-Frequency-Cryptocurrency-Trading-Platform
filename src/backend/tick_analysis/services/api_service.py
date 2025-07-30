import ccxt
import aiohttp
import asyncio
from typing import Dict, Any, Optional
import logging
from datetime import datetime
import json

from ..config.api_keys import get_api_config

logger = logging.getLogger(__name__)

class APIService:
    def __init__(self):
        self.config = get_api_config()
        self._setup_clients()
        
    def _setup_clients(self):
        """Initialize API clients with the configured keys."""
        try:
            # Setup Binance client
            self.binance = ccxt.binance({
                'apiKey': self.config.BINANCE_API_KEY.get_secret_value(),
                'secret': self.config.BINANCE_API_SECRET.get_secret_value(),
                'enableRateLimit': True
            })
            
            # Setup Polygon.io headers
            self.polygon_headers = {
                'Authorization': f'Bearer {self.config.POLYGON_API_KEY.get_secret_value()}'
            }
            
            # Setup CryptoCompare headers
            self.cryptocompare_headers = {
                'authorization': f'Apikey {self.config.CRYPTOCOMPARE_API_KEY.get_secret_value()}'
            }
            
            logger.info("API clients initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing API clients: {str(e)}")
            raise
            
    async def validate_connections(self) -> Dict[str, bool]:
        """Validate all API connections."""
        results = {
            'binance': False,
            'polygon': False,
            'cryptocompare': False
        }
        
        try:
            # Test Binance connection
            await asyncio.to_thread(self.binance.fetch_balance)
            results['binance'] = True
        except Exception as e:
            logger.error(f"Binance connection error: {str(e)}")
            
        try:
            # Test Polygon.io connection
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    'https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/2023-01-09/2023-01-09',
                    headers=self.polygon_headers
                ) as response:
                    if response.status == 200:
                        results['polygon'] = True
        except Exception as e:
            logger.error(f"Polygon.io connection error: {str(e)}")
            
        try:
            # Test CryptoCompare connection
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    'https://min-api.cryptocompare.com/data/price?fsym=BTC&tsyms=USD',
                    headers=self.cryptocompare_headers
                ) as response:
                    if response.status == 200:
                        results['cryptocompare'] = True
        except Exception as e:
            logger.error(f"CryptoCompare connection error: {str(e)}")
            
        return results
        
    async def get_market_data(self, symbol: str, timeframe: str = '1m') -> Dict[str, Any]:
        """Get market data from multiple sources."""
        data = {}
        
        try:
            # Get data from Binance
            binance_data = await asyncio.to_thread(
                self.binance.fetch_ohlcv,
                symbol,
                timeframe,
                limit=100
            )
            data['binance'] = binance_data
        except Exception as e:
            logger.error(f"Error fetching Binance data: {str(e)}")
            
        try:
            # Get data from Polygon.io
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f'https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/{timeframe}/2023-01-09/2023-01-09',
                    headers=self.polygon_headers
                ) as response:
                    if response.status == 200:
                        data['polygon'] = await response.json()
        except Exception as e:
            logger.error(f"Error fetching Polygon.io data: {str(e)}")
            
        return data
        
    async def get_crypto_price(self, symbol: str) -> Optional[float]:
        """Get current cryptocurrency price."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f'https://min-api.cryptocompare.com/data/price?fsym={symbol}&tsyms=USD',
                    headers=self.cryptocompare_headers
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('USD')
        except Exception as e:
            logger.error(f"Error fetching crypto price: {str(e)}")
        return None
        
    async def get_stock_price(self, symbol: str) -> Optional[float]:
        """Get current stock price."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f'https://api.polygon.io/v2/last/trade/{symbol}',
                    headers=self.polygon_headers
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('last', {}).get('price')
        except Exception as e:
            logger.error(f"Error fetching stock price: {str(e)}")
        return None

# Create a singleton instance
api_service = APIService()

def get_api_service() -> APIService:
    """Get the API service singleton instance."""
    return api_service 