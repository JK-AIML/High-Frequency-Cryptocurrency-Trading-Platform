"""
Tests for the WebSocket client.
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from tick_analysis.core.data.websocket_client import WebSocketClient


@pytest.fixture
def websocket_client():
    """Create a WebSocket client instance for testing."""
    return WebSocketClient(exchange="binance", symbols=["BTC/USDT"])


@pytest.mark.asyncio
async def test_connect_success(websocket_client):
    """Test successful WebSocket connection."""
    mock_ws = AsyncMock()
    mock_ws.__aenter__.return_value = mock_ws

    with patch("websockets.connect", return_value=mock_ws) as mock_connect:
        await websocket_client.connect()

        assert websocket_client.websocket is not None
        mock_connect.assert_called_once()


@pytest.mark.asyncio
async def test_connect_already_connected(websocket_client, caplog):
    """Test connecting when already connected."""
    websocket_client.websocket = AsyncMock()

    await websocket_client.connect()

    assert "Already connected to WebSocket" in caplog.text


@pytest.mark.asyncio
async def test_disconnect(websocket_client):
    """Test disconnecting from the WebSocket."""
    mock_ws = AsyncMock()
    websocket_client.websocket = mock_ws

    await websocket_client.disconnect()

    mock_ws.close.assert_awaited_once()
    assert websocket_client.websocket is None


@pytest.mark.asyncio
async def test_subscribe(websocket_client):
    """Test subscribing to a symbol."""
    with patch.object(websocket_client, "send") as mock_send:
        websocket_client.subscribe("BTC/USDT")
        assert "BTC/USDT" in websocket_client.symbols


@pytest.mark.asyncio
async def test_unsubscribe(websocket_client):
    """Test unsubscribing from a symbol."""
    websocket_client.symbols = ["BTC/USDT", "ETH/USDT"]
    websocket_client.unsubscribe("ETH/USDT")
    assert "ETH/USDT" not in websocket_client.symbols
    assert "BTC/USDT" in websocket_client.symbols


@pytest.mark.asyncio
async def test_handle_stream(websocket_client):
    """Test handling incoming WebSocket messages."""
    mock_callback = AsyncMock()
    websocket_client.callbacks["trade"] = [mock_callback]

    message = json.dumps(
        {"e": "trade", "s": "BTCUSDT", "p": "50000.00", "q": "0.1", "T": 1620000000000}
    )

    await websocket_client._handle_message(message)
    mock_callback.assert_awaited_once()


@pytest.mark.asyncio
async def test_handle_stream_json_error(websocket_client, caplog):
    """Test handling invalid JSON in WebSocket messages."""
    await websocket_client._handle_message("invalid json")
    assert "Error parsing WebSocket message" in caplog.text


@pytest.mark.asyncio
async def test_start_stop(websocket_client):
    """Test starting and stopping the WebSocket client."""
    mock_ws = AsyncMock()
    mock_ws.__aenter__.return_value = mock_ws

    with patch("websockets.connect", return_value=mock_ws) as mock_connect:
        # Mock the _run method to avoid running the infinite loop
        with patch.object(websocket_client, "_run", new_callable=AsyncMock) as mock_run:
            # Start the client
            await websocket_client.start()
            await asyncio.sleep(0.1)  # Allow the start coroutine to run

            assert websocket_client.running
            assert websocket_client.websocket is not None

            # Stop the client
            await websocket_client.stop()
            assert not websocket_client.running
            assert websocket_client.websocket is None


@pytest.mark.asyncio
async def test_subscribe_to_ticker(websocket_client):
    """Test subscribing to ticker updates."""
    with patch.object(websocket_client, "subscribe_to_stream") as mock_subscribe:
        await websocket_client.subscribe_to_ticker()
        mock_subscribe.assert_called_once_with("ticker")


@pytest.mark.asyncio
async def test_subscribe_to_depth(websocket_client):
    """Test subscribing to order book depth updates."""
    with patch.object(websocket_client, "subscribe_to_stream") as mock_subscribe:
        await websocket_client.subscribe_to_depth()
        mock_subscribe.assert_called_once_with("depth")


@pytest.mark.asyncio
async def test_subscribe_to_trades(websocket_client):
    """Test subscribing to trade updates."""
    with patch.object(websocket_client, "subscribe_to_stream") as mock_subscribe:
        await websocket_client.subscribe_to_trades()
        mock_subscribe.assert_called_once_with("trade")


@pytest.mark.asyncio
async def test_subscribe_to_kline(websocket_client):
    """Test subscribing to kline/candlestick updates."""
    with patch.object(websocket_client, "subscribe_to_stream") as mock_subscribe:
        await websocket_client.subscribe_to_kline(interval="1m")
        mock_subscribe.assert_called_once_with("kline_1m")
