import pytest
from unittest.mock import Mock, patch
from src.tick_analysis.services.api_service import APIService
from datetime import datetime

@pytest.fixture
def mock_binance():
    """Mock Binance API client."""
    with patch('src.tick_analysis.services.api_service.Client') as mock:
        mock_client = Mock()
        mock_client.get_klines.return_value = [
            [
                1625097600000,  # Open time
                "35000.00",     # Open
                "36000.00",     # High
                "34000.00",     # Low
                "35500.00",     # Close
                "100.00",       # Volume
                1625097899999,  # Close time
                "3500000.00",   # Quote asset volume
                100,            # Number of trades
                "50.00",        # Taker buy base asset volume
                "1750000.00",   # Taker buy quote asset volume
                "0"             # Ignore
            ]
        ]
        mock.return_value = mock_client
        yield mock_client

@pytest.fixture
def mock_polygon():
    """Mock Polygon.io API client."""
    with patch('src.tick_analysis.services.api_service.RESTClient') as mock:
        mock_client = Mock()
        mock_client.get_aggs.return_value = [
            Mock(
                timestamp=1625097600000,
                open=350.00,
                high=360.00,
                low=340.00,
                close=355.00,
                volume=1000,
                transactions=100
            )
        ]
        mock.return_value = mock_client
        yield mock_client

@pytest.fixture
def mock_cryptocompare():
    """Mock CryptoCompare API client."""
    with patch('src.tick_analysis.services.api_service.requests.get') as mock:
        mock_response = Mock()
        mock_response.json.return_value = {
            "Data": {
                "Data": [
                    {
                        "time": 1625097600,
                        "open": 35000.00,
                        "high": 36000.00,
                        "low": 34000.00,
                        "close": 35500.00,
                        "volumefrom": 100.00,
                        "volumeto": 3500000.00
                    }
                ]
            }
        }
        mock.return_value = mock_response
        yield mock

@pytest.mark.asyncio
async def test_get_market_data_binance(mock_binance):
    """Test getting market data from Binance."""
    api_service = APIService()
    data = await api_service.get_market_data("BTC/USD", "1m", "binance")
    
    assert data is not None
    assert "binance" in data
    binance_data = data["binance"]
    assert len(binance_data) > 0
    assert all(key in binance_data[0] for key in ["timestamp", "open", "high", "low", "close", "volume"])

@pytest.mark.asyncio
async def test_get_market_data_polygon(mock_polygon):
    """Test getting market data from Polygon.io."""
    api_service = APIService()
    data = await api_service.get_market_data("AAPL", "1m", "polygon")
    
    assert data is not None
    assert "polygon" in data
    polygon_data = data["polygon"]
    assert len(polygon_data) > 0
    assert all(key in polygon_data[0] for key in ["timestamp", "open", "high", "low", "close", "volume"])

@pytest.mark.asyncio
async def test_get_market_data_cryptocompare(mock_cryptocompare):
    """Test getting market data from CryptoCompare."""
    api_service = APIService()
    data = await api_service.get_market_data("BTC/USD", "1m", "cryptocompare")
    
    assert data is not None
    assert "cryptocompare" in data
    cryptocompare_data = data["cryptocompare"]
    assert len(cryptocompare_data) > 0
    assert all(key in cryptocompare_data[0] for key in ["timestamp", "open", "high", "low", "close", "volume"])

@pytest.mark.asyncio
async def test_validate_connections(mock_binance, mock_polygon, mock_cryptocompare):
    """Test API connection validation."""
    api_service = APIService()
    results = await api_service.validate_connections()
    
    assert isinstance(results, dict)
    assert "binance" in results
    assert "polygon" in results
    assert "cryptocompare" in results
    
    for source, status in results.items():
        assert isinstance(status, bool)

@pytest.mark.asyncio
async def test_error_handling(mock_binance):
    """Test error handling in market data retrieval."""
    mock_binance.get_klines.side_effect = Exception("API Error")
    
    api_service = APIService()
    data = await api_service.get_market_data("BTC/USD", "1m", "binance")
    
    assert data is not None
    assert "binance" in data
    assert "error" in data["binance"]
    assert data["binance"]["error"] == "API Error" 