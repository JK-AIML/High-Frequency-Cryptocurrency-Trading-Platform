import unittest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from tick_analysis.data.collector import DataCollector


class TestDataCollector(unittest.TestCase):
    def setUp(self):
        self.collector = DataCollector()

    @patch("tick_analysis.data.collectors.cryptocompare_collector")
    def test_get_binance_data(self, mock_binance):
        # Mock Binance response
        mock_data = [
            [1628512800000, 47000.0, 47050.0, 46950.0, 47025.0, 100.0],
            [1628512860000, 47025.0, 47075.0, 47000.0, 47050.0, 150.0],
        ]
        mock_binance.fetch_ohlcv.return_value = mock_data

        # Test data collection
        data = self.collector._get_binance_data(
            symbol="BTC/USDT",
            timeframe="1m",
            start_time=datetime(2021, 8, 9),
            end_time=datetime(2021, 8, 10),
        )

        # Verify data structure
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(len(data), 2)
        self.assertTrue(
            all(
                col in data.columns
                for col in ["timestamp", "open", "high", "low", "close", "volume"]
            )
        )

    @patch("tick_analysis.data.collectors.cryptocompare_collector.requests.get")
    def test_get_polygon_data(self, mock_get):
        # Mock Polygon response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "OK",
            "results": [
                {
                    "t": 1628512800000,
                    "o": 47000.0,
                    "h": 47050.0,
                    "l": 46950.0,
                    "c": 47025.0,
                    "v": 100.0,
                },
                {
                    "t": 1628512860000,
                    "o": 47025.0,
                    "h": 47075.0,
                    "l": 47000.0,
                    "c": 47050.0,
                    "v": 150.0,
                },
            ],
        }
        mock_get.return_value = mock_response

        # Test data collection
        data = self.collector._get_polygon_data(
            symbol="BTC/USDT",
            timeframe="1m",
            start_time=datetime(2021, 8, 9),
            end_time=datetime(2021, 8, 10),
        )

        # Verify data structure
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(len(data), 2)
        self.assertTrue(
            all(
                col in data.columns
                for col in ["timestamp", "open", "high", "low", "close", "volume"]
            )
        )

    @patch("tick_analysis.data.collectors.cryptocompare_collector.requests.get")
    def test_get_cryptocompare_data(self, mock_get):
        # Mock CryptoCompare response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "Response": "Success",
            "Data": {
                "Data": [
                    {
                        "time": 1628512800,
                        "open": 47000.0,
                        "high": 47050.0,
                        "low": 46950.0,
                        "close": 47025.0,
                        "volumefrom": 100.0,
                    },
                    {
                        "time": 1628512860,
                        "open": 47025.0,
                        "high": 47075.0,
                        "low": 47000.0,
                        "close": 47050.0,
                        "volumefrom": 150.0,
                    },
                ]
            },
        }
        mock_get.return_value = mock_response

        # Test data collection
        data = self.collector._get_cryptocompare_data(
            symbol="BTC/USDT",
            timeframe="1m",
            start_time=datetime(2021, 8, 9),
            end_time=datetime(2021, 8, 10),
        )

        # Verify data structure
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(len(data), 2)
        self.assertTrue(
            all(
                col in data.columns
                for col in ["timestamp", "open", "high", "low", "close", "volume"]
            )
        )

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            self.collector.get_tick_data("BTC/USDT", timeframe="invalid")

        with self.assertRaises(ValueError):
            self.collector.get_tick_data("BTC/USDT", source="invalid")

        with self.assertRaises(ValueError):
            self.collector.get_tick_data(
                "BTC/USDT",
                start_time=datetime(2021, 8, 10),
                end_time=datetime(2021, 8, 9),
            )


if __name__ == "__main__":
    unittest.main()
