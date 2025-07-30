"""
Tests for the EnhancedDataCollector class.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio
import os

# Add project root to path
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from tick_analysis.data.enhanced_collector import EnhancedDataCollector, RateLimiter

# Test data
SAMPLE_OHLCV = [
    [1614556800000, 50000.0, 51000.0, 49000.0, 50500.0, 1000.0],
    [1614560400000, 50500.0, 51500.0, 50000.0, 51000.0, 1200.0],
    [1614564000000, 51000.0, 52000.0, 50500.0, 51500.0, 1500.0],
]


@pytest.fixture
def mock_ccxt_exchange():
    """Create a mock CCXT exchange."""
    exchange = MagicMock()
    exchange.fetch_ohlcv = AsyncMock(return_value=SAMPLE_OHLCV)
    return exchange


@pytest.fixture
def collector():
    """Create a test instance of EnhancedDataCollector."""
    return EnhancedDataCollector()


@pytest.mark.asyncio
async def test_get_ohlcv_success(collector, mock_ccxt_exchange):
    """Test successful OHLCV data retrieval."""
    with patch.dict(collector.exchanges, {"binance": mock_ccxt_exchange}):
        df = await collector.get_ohlcv("BTC/USDT", "1h", limit=3, exchange="binance")

        # Verify the exchange method was called correctly
        mock_ccxt_exchange.fetch_ohlcv.assert_awaited_once_with(
            symbol="BTC/USDT", timeframe="1h", since=None, limit=3
        )

        # Verify the returned DataFrame
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert df.index.name == "timestamp"
        assert not df.isnull().any().any()


@pytest.mark.asyncio
async def test_get_ohlcv_network_retry(collector, mock_ccxt_exchange):
    """Test retry behavior on network errors."""
    # Make the first call fail with a network error
    mock_ccxt_exchange.fetch_ohlcv.side_effect = [
        Exception("Network error"),  # First attempt fails
        SAMPLE_OHLCV,  # Second attempt succeeds
    ]

    with patch.dict(collector.exchanges, {"binance": mock_ccxt_exchange}):
        df = await collector.get_ohlcv("BTC/USDT", "1h", limit=3, exchange="binance")

        # Should have retried once
        assert mock_ccxt_exchange.fetch_ohlcv.await_count == 2
        assert len(df) == 3


@pytest.mark.asyncio
async def test_get_multiple_ohlcv(collector, mock_ccxt_exchange):
    """Test fetching OHLCV data for multiple symbols."""
    with patch.dict(collector.exchanges, {"binance": mock_ccxt_exchange}):
        symbols = ["BTC/USDT", "ETH/USDT"]
        results = await collector.get_multiple_ohlcv(
            symbols=symbols, timeframe="1h", limit=3, exchange="binance"
        )

        # Verify results
        assert set(results.keys()) == set(symbols)
        for symbol in symbols:
            assert len(results[symbol]) == 3


@pytest.mark.asyncio
async def test_symbol_normalization(collector, mock_ccxt_exchange):
    """Test symbol normalization."""
    with patch.dict(collector.exchanges, {"binance": mock_ccxt_exchange}):
        # Test with hyphenated symbol
        await collector.get_ohlcv("BTC-USDT", "1h", exchange="binance")
        mock_ccxt_exchange.fetch_ohlcv.assert_awaited_once()

        # Verify symbol was normalized
        args, kwargs = mock_ccxt_exchange.fetch_ohlcv.await_args
        assert kwargs["symbol"] == "BTC/USDT"


def test_rate_limiter():
    """Test the rate limiter configuration."""
    limiter = RateLimiter(max_retries=3, initial_delay=1.0, max_delay=10.0)

    # Verify the retry strategy is configured correctly
    assert limiter.retry_strategy.total == 3
    assert limiter.retry_strategy.backoff_factor == 2.0
    assert limiter.retry_strategy.status_forcelist == [429, 500, 502, 503, 504]

    # Verify the session is configured with the retry strategy
    assert limiter.session.adapters["https://"].max_retries == limiter.retry_strategy


@pytest.mark.asyncio
async def test_exchange_initialization():
    """Test exchange initialization with environment variables."""
    with patch.dict(
        os.environ, {"BINANCE_API_KEY": "test_key", "BINANCE_API_SECRET": "test_secret"}
    ):
        collector = EnhancedDataCollector()
        assert "binance" in collector.exchanges
        assert collector.exchanges["binance"].apiKey == "test_key"
        assert collector.exchanges["binance"].secret == "test_secret"
