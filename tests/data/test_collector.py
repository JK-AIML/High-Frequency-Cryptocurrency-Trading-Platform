"""
Tests for the DataCollector class.
"""

import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone, timedelta
from decimal import Decimal
import pandas as pd
import numpy as np
import aiohttp
import asyncio

from tick_analysis.data.collector import DataCollector
from tick_analysis.data.models import Timeframe, Candle


class TestDataCollector(unittest.TestCase):
    """Test cases for DataCollector."""

    def setUp(self):
        """Set up test fixtures."""
        self.symbol = "BTC/USDT"
        self.timeframe = Timeframe.ONE_HOUR
        self.start_date = datetime(2023, 1, 1, tzinfo=timezone.utc)
        self.end_date = datetime(2023, 1, 2, tzinfo=timezone.utc)

        # Mock API responses
        self.mock_candles = [
            {
                "timestamp": int(
                    (self.start_date + timedelta(hours=i)).timestamp() * 1000
                ),
                "open": 50000 + (i * 10),
                "high": 50100 + (i * 10),
                "low": 49900 + (i * 10),
                "close": 50050 + (i * 10),
                "volume": 1000 + (i * 10),
            }
            for i in range(24)  # 24 hours of data
        ]

        # Initialize collector with mock session
        self.session = MagicMock(spec=aiohttp.ClientSession)
        self.collector = DataCollector(
            api_key="test_key", api_secret="test_secret", session=self.session
        )

    async def async_setup(self):
        """Async setup for tests."""
        self.session = aiohttp.ClientSession()
        self.collector = DataCollector(
            api_key="test_key", api_secret="test_secret", session=self.session
        )

    async def async_teardown(self):
        """Clean up async resources."""
        if hasattr(self, "session") and not self.session.closed:
            await self.session.close()

    def run_async(self, coro):
        """Run async test method."""
        return asyncio.get_event_loop().run_until_complete(coro)

    @patch("aiohttp.ClientSession.get")
    def test_fetch_historical_data(self, mock_get):
        """Test fetching historical data."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json.return_value = self.mock_candles

        # Setup mock context manager
        mock_get.return_value.__aenter__.return_value = mock_response

        # Run test
        async def test():
            await self.async_setup()
            try:
                df = await self.collector.fetch_historical_data(
                    symbol=self.symbol,
                    timeframe=self.timeframe,
                    start_date=self.start_date,
                    end_date=self.end_date,
                )

                # Assertions
                self.assertIsInstance(df, pd.DataFrame)
                self.assertEqual(len(df), 24)  # 24 hours of data
                self.assertIn("open", df.columns)
                self.assertIn("high", df.columns)
                self.assertIn("low", df.columns)
                self.assertIn("close", df.columns)
                self.assertIn("volume", df.columns)

                # Check index is datetime
                self.assertIsInstance(df.index, pd.DatetimeIndex)

            finally:
                await self.async_teardown()

        self.run_async(test())

    @patch("aiohttp.ClientSession.get")
    def test_fetch_historical_data_error(self, mock_get):
        """Test error handling in fetch_historical_data."""
        # Setup mock to raise an exception
        mock_get.side_effect = aiohttp.ClientError("API Error")

        # Run test
        async def test():
            await self.async_setup()
            try:
                with self.assertRaises(aiohttp.ClientError):
                    await self.collector.fetch_historical_data(
                        symbol=self.symbol,
                        timeframe=self.timeframe,
                        start_date=self.start_date,
                        end_date=self.end_date,
                    )
            finally:
                await self.async_teardown()

        self.run_async(test())

    @patch("aiohttp.ClientSession.ws_connect")
    def test_websocket_connection(self, mock_ws_connect):
        """Test WebSocket connection setup."""
        # Setup mock WebSocket
        mock_ws = MagicMock()
        mock_ws_connect.return_value.__aenter__.return_value = mock_ws

        # Run test
        async def test():
            await self.async_setup()
            try:
                await self.collector.connect_websocket()
                self.assertTrue(self.collector.websocket_connected)

                # Test sending a message
                await self.collector.subscribe_to_symbol(self.symbol)
                mock_ws.send_json.assert_called_once()

                # Test disconnection
                await self.collector.disconnect_websocket()
                self.assertFalse(self.collector.websocket_connected)

            finally:
                await self.async_teardown()

        self.run_async(test())

    def test_process_candle(self) -> None:
        """Test candle processing."""
        # Create a raw candle
        raw_candle = self.mock_candles[0]

        # Process the candle
        candle = self.collector._process_candle(raw_candle, self.symbol, self.timeframe)

        # Assertions
        self.assertIsInstance(candle, Candle)
        self.assertEqual(candle.symbol, self.symbol)
        self.assertEqual(candle.timeframe, self.timeframe)
        self.assertEqual(candle.open, Decimal(str(raw_candle["open"])))
        self.assertEqual(candle.high, Decimal(str(raw_candle["high"])))
        self.assertEqual(candle.low, Decimal(str(raw_candle["low"])))
        self.assertEqual(candle.close, Decimal(str(raw_candle["close"])))
        self.assertEqual(candle.volume, Decimal(str(raw_candle["volume"])))
        self.assertEqual(
            candle.timestamp,
            datetime.fromtimestamp(raw_candle["timestamp"] / 1000, tz=timezone.utc),
        )


if __name__ == "__main__":
    unittest.main()
