"""
Tests for the DataManager class.
"""

import pytest
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, AsyncMock
import shutil
from pathlib import Path

# Add project root to path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from tick_analysis.data.data_manager import DataManager

# Sample test data
SAMPLE_OHLCV = [
    [1614556800000, 50000.0, 51000.0, 49000.0, 50500.0, 1000.0],
    [1614560400000, 50500.0, 51500.0, 50000.0, 51000.0, 1200.0],
    [1614564000000, 51000.0, 52000.0, 50500.0, 51500.0, 1500.0],
]

SAMPLE_DF = pd.DataFrame(
    SAMPLE_OHLCV, columns=["timestamp", "open", "high", "low", "close", "volume"]
)
SAMPLE_DF["timestamp"] = pd.to_datetime(SAMPLE_DF["timestamp"], unit="ms")
SAMPLE_DF.set_index("timestamp", inplace=True)


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create a temporary cache directory for testing."""
    cache_dir = tmp_path / "test_cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def data_manager(temp_cache_dir):
    """Create a DataManager instance with a temporary cache directory."""
    return DataManager(cache_dir=str(temp_cache_dir))


@pytest.mark.asyncio
async def test_get_ohlcv_success(data_manager, temp_cache_dir):
    """Test successful OHLCV data retrieval with caching."""
    # Mock the collector's get_ohlcv method
    with patch.object(
        data_manager.collector,
        "get_ohlcv",
        new_callable=AsyncMock,
        return_value=SAMPLE_DF,
    ):

        # First call - should fetch from API and cache
        start_time = datetime(2023, 1, 1)
        end_time = datetime(2023, 1, 2)
        df = await data_manager.get_ohlcv(
            "BTC/USDT", "1h", start_time, end_time, use_cache=True
        )

        # Verify the result
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "open" in df.columns

        # Verify cache was created
        cache_file = temp_cache_dir / "BTC_USDT_1h_1672531200_1672617600.parquet"
        assert cache_file.exists()

        # Second call - should load from cache
        with patch.object(
            data_manager.collector,
            "get_ohlcv",
            side_effect=Exception("Should not be called"),
        ):
            df_cached = await data_manager.get_ohlcv(
                "BTC/USDT", "1h", start_time, end_time, use_cache=True
            )
            pd.testing.assert_frame_equal(df, df_cached)


@pytest.mark.asyncio
async def test_get_multiple_ohlcv(data_manager):
    """Test fetching OHLCV data for multiple symbols."""
    # Mock the collector's get_ohlcv method
    with patch.object(
        data_manager.collector,
        "get_ohlcv",
        new_callable=AsyncMock,
        return_value=SAMPLE_DF,
    ):

        symbols = ["BTC/USDT", "ETH/USDT"]
        start_time = datetime(2023, 1, 1)
        end_time = datetime(2023, 1, 2)

        data = await data_manager.get_multiple_ohlcv(
            symbols=symbols, timeframe="1h", start_time=start_time, end_time=end_time
        )

        assert set(data.keys()) == set(symbols)
        for symbol in symbols:
            assert len(data[symbol]) == 3


@pytest.mark.asyncio
async def test_cache_invalidation(data_manager, temp_cache_dir):
    """Test that cache is properly invalidated when data changes."""
    # First, fetch and cache some data
    with patch.object(
        data_manager.collector,
        "get_ohlcv",
        new_callable=AsyncMock,
        return_value=SAMPLE_DF,
    ):

        await data_manager.get_ohlcv("BTC/USDT", "1h", use_cache=True)

        # Verify cache was created
        cache_files = list(temp_cache_dir.glob("*.parquet"))
        assert len(cache_files) == 1

        # Now change the return value
        modified_df = SAMPLE_DF.copy()
        modified_df["close"] *= 1.1  # 10% price increase

        with patch.object(
            data_manager.collector,
            "get_ohlcv",
            new_callable=AsyncMock,
            return_value=modified_df,
        ):

            # Force refresh by not using cache
            df = await data_manager.get_ohlcv("BTC/USDT", "1h", use_cache=False)

            # Verify we got the modified data
            assert df["close"].iloc[0] == pytest.approx(50500.0 * 1.1)


def test_available_symbols(data_manager):
    """Test getting available symbols."""
    symbols = data_manager.get_available_symbols()
    assert isinstance(symbols, list)
    assert len(symbols) > 0
    assert all(isinstance(s, str) for s in symbols)


def test_available_timeframes(data_manager):
    """Test getting available timeframes."""
    timeframes = data_manager.get_available_timeframes()
    assert isinstance(timeframes, list)
    assert len(timeframes) > 0
    assert all(isinstance(tf, str) for tf in timeframes)
