"""
WebSocket client for real-time market data streaming.
"""
import asyncio
import json
import logging
from typing import Dict, List, Callable, Optional, Any
import websockets
from websockets.client import WebSocketClientProtocol
from dataclasses import dataclass, asdict
from datetime import datetime
import time
import signal
import ssl

logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Market data tick representation."""
    symbol: str
    timestamp: float
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    exchange: str
    data_type: str = "tick"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'bid': self.bid,
            'ask': self.ask,
            'bid_size': self.bid_size,
            'ask_size': self.ask_size,
            'exchange': self.exchange,
            'data_type': self.data_type,
            'received_at': time.time()
        }

class WebSocketClient:
    """
    WebSocket client for streaming market data.
    
    Features:
    - Auto-reconnect with exponential backoff
    - Multiple message handlers
    - Subscription management
    - Heartbeat/ping-pong
    - Error handling and recovery
    """
    
    def __init__(
        self,
        url: str,
        api_key: Optional[str] = None,
        ping_interval: int = 30,
        max_reconnect_attempts: int = 10,
        reconnect_delay: float = 1.0,
        max_reconnect_delay: float = 60.0,
        ssl_context: Optional[ssl.SSLContext] = None
    ):
        """
        Initialize WebSocket client.
        
        Args:
            url: WebSocket server URL (wss:// or ws://)
            api_key: Optional API key for authentication
            ping_interval: Ping interval in seconds
            max_reconnect_attempts: Maximum reconnection attempts
            reconnect_delay: Initial reconnection delay in seconds
            max_reconnect_delay: Maximum reconnection delay in seconds
            ssl_context: Optional SSL context
        """
        self.url = url
        self.api_key = api_key
        self.ping_interval = ping_interval
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_delay = max_reconnect_delay
        self.ssl_context = ssl_context
        
        self.websocket: Optional[WebSocketClientProtocol] = None
        self.connected = False
        self.reconnect_attempts = 0
        self.last_ping = 0
        self.last_pong = 0
        self.subscriptions: List[Dict[str, Any]] = []
        self.handlers: List[Callable[[Dict[str, Any]], None]] = []
        self.running = False
        self.task: Optional[asyncio.Task] = None
        
        # Register signal handlers for clean shutdown
        try:
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, self.stop)
        except (NotImplementedError, RuntimeError):
            # Signals not supported on this platform
            pass
    
    async def connect(self) -> None:
        """Connect to WebSocket server."""
        if self.connected:
            return
            
        headers = {}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        
        while self.reconnect_attempts < self.max_reconnect_attempts and not self.connected:
            try:
                logger.info(f"Connecting to {self.url}...")
                self.websocket = await websockets.connect(
                    self.url,
                    extra_headers=headers,
                    ssl=self.ssl_context,
                    ping_interval=None,  # We'll handle ping/pong ourselves
                    close_timeout=1,
                    max_size=2**25,  # 32MB
                    max_queue=1024
                )
                self.connected = True
                self.reconnect_attempts = 0
                self.reconnect_delay = 1.0
                logger.info("WebSocket connected")
                
                # Start reader and writer tasks
                self.running = True
                self.task = asyncio.create_task(self._run())
                
                # Resubscribe to channels
                if self.subscriptions:
                    await self._resubscribe()
                
            except Exception as e:
                self.connected = False
                self.reconnect_attempts += 1
                delay = min(self.reconnect_delay * (2 ** (self.reconnect_attempts - 1)), 
                           self.max_reconnect_delay)
                logger.error(
                    f"Connection failed (attempt {self.reconnect_attempts}/"
                    f"{self.max_reconnect_attempts}): {str(e)}. "
                    f"Retrying in {delay:.1f}s..."
                )
                await asyncio.sleep(delay)
        
        if not self.connected:
            raise ConnectionError(
                f"Failed to connect after {self.reconnect_attempts} attempts"
            )
    
    async def _run(self) -> None:
        """Run the WebSocket client."""
        if not self.websocket:
            return
            
        last_ping = time.time()
        
        try:
            while self.running and self.connected:
                try:
                    # Send ping if needed
                    now = time.time()
                    if now - last_ping >= self.ping_interval:
                        await self._send_ping()
                        last_ping = now
                    
                    # Set a timeout for the receive operation
                    try:
                        message = await asyncio.wait_for(
                            self.websocket.recv(),
                            timeout=1.0
                        )
                        await self._handle_message(message)
                    except asyncio.TimeoutError:
                        # Check connection status
                        if now - self.last_pong > self.ping_interval * 3:
                            logger.warning("No pong received, reconnecting...")
                            await self._reconnect()
                            continue
                    
                except websockets.exceptions.ConnectionClosed as e:
                    logger.error(f"WebSocket connection closed: {e}")
                    await self._reconnect()
                    
                except Exception as e:
                    logger.error(f"Error in WebSocket loop: {str(e)}", exc_info=True)
                    await asyncio.sleep(1)
                    
        except asyncio.CancelledError:
            logger.info("WebSocket task cancelled")
        except Exception as e:
            logger.error(f"Fatal error in WebSocket loop: {str(e)}", exc_info=True)
        finally:
            await self.disconnect()
    
    async def _handle_message(self, message: str) -> None:
        """Handle incoming WebSocket message."""
        try:
            if isinstance(message, bytes):
                # Handle binary messages (e.g., protocol buffers)
                data = self._parse_binary_message(message)
            else:
                # Handle text messages (JSON)
                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON message: {message[:200]}...")
                    return
            
            # Handle ping/pong
            if data.get('type') == 'ping':
                await self._send_pong()
                return
            elif data.get('type') == 'pong':
                self.last_pong = time.time()
                return
            
            # Call registered handlers
            for handler in self.handlers:
                try:
                    handler(data)
                except Exception as e:
                    logger.error(f"Error in message handler: {str(e)}", exc_info=True)
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}", exc_info=True)
    
    def _parse_binary_message(self, message: bytes) -> Dict[str, Any]:
        """Parse binary message (override in subclasses)."""
        # Default implementation assumes JSON-encoded bytes
        try:
            return json.loads(message.decode('utf-8'))
        except UnicodeDecodeError:
            logger.warning(f"Could not decode binary message: {message[:100]}...")
            return {'type': 'binary', 'data': message.hex()}
    
    async def _send_ping(self) -> None:
        """Send ping message."""
        if not self.connected or not self.websocket:
            return
            
        try:
            self.last_ping = time.time()
            await self.websocket.ping()
        except Exception as e:
            logger.error(f"Error sending ping: {str(e)}")
            await self._reconnect()
    
    async def _send_pong(self) -> None:
        """Send pong message."""
        if not self.connected or not self.websocket:
            return
            
        try:
            await self.websocket.pong()
        except Exception as e:
            logger.error(f"Error sending pong: {str(e)}")
            await self._reconnect()
    
    async def _reconnect(self) -> None:
        """Handle reconnection."""
        self.connected = False
        await self.disconnect()
        await asyncio.sleep(self.reconnect_delay)
        await self.connect()
    
    async def _resubscribe(self) -> None:
        """Resubscribe to channels after reconnection."""
        if not self.subscriptions:
            return
            
        logger.info(f"Resubscribing to {len(self.subscriptions)} channels...")
        for sub in self.subscriptions:
            try:
                await self.subscribe(**sub)
            except Exception as e:
                logger.error(f"Error resubscribing to {sub}: {str(e)}")
    
    async def subscribe(
        self,
        channel: str,
        symbols: List[str],
        **kwargs
    ) -> None:
        """
        Subscribe to a channel.
        
        Args:
            channel: Channel name (e.g., 'ticker', 'trades', 'orderbook')
            symbols: List of symbols to subscribe to
            **kwargs: Additional subscription parameters
        """
        if not self.connected or not self.websocket:
            raise ConnectionError("Not connected to WebSocket server")
            
        subscription = {
            'type': 'subscribe',
            'channel': channel,
            'symbols': symbols,
            **kwargs
        }
        
        # Store subscription for reconnection
        self.subscriptions.append({
            'channel': channel,
            'symbols': symbols,
            **kwargs
        })
        
        await self._send_json(subscription)
    
    async def unsubscribe(
        self,
        channel: str,
        symbols: Optional[List[str]] = None,
        **kwargs
    ) -> None:
        """
        Unsubscribe from a channel.
        
        Args:
            channel: Channel name
            symbols: Optional list of symbols to unsubscribe from
            **kwargs: Additional parameters to match subscription
        """
        if not self.connected or not self.websocket:
            raise ConnectionError("Not connected to WebSocket server")
            
        # Remove from subscriptions
        self.subscriptions = [
            sub for sub in self.subscriptions
            if not (
                sub['channel'] == channel and
                (symbols is None or sub.get('symbols') == symbols) and
                all(sub.get(k) == v for k, v in kwargs.items())
            )
        ]
        
        unsubscribe_msg = {
            'type': 'unsubscribe',
            'channel': channel,
            'symbols': symbols or [],
            **kwargs
        }
        
        await self._send_json(unsubscribe_msg)
    
    async def _send_json(self, data: Dict[str, Any]) -> None:
        """Send JSON data through WebSocket."""
        if not self.connected or not self.websocket:
            raise ConnectionError("Not connected to WebSocket server")
            
        try:
            await self.websocket.send(json.dumps(data))
        except Exception as e:
            logger.error(f"Error sending message: {str(e)}")
            await self._reconnect()
            raise
    
    def add_handler(self, handler: Callable[[Dict[str, Any]], None]) -> None:
        """Add a message handler."""
        if handler not in self.handlers:
            self.handlers.append(handler)
    
    def remove_handler(self, handler: Callable[[Dict[str, Any]], None]) -> None:
        """Remove a message handler."""
        if handler in self.handlers:
            self.handlers.remove(handler)
    
    async def disconnect(self) -> None:
        """Disconnect from WebSocket server."""
        self.running = False
        self.connected = False
        
        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception as e:
                logger.error(f"Error closing WebSocket: {str(e)}")
            finally:
                self.websocket = None
    
    def stop(self) -> None:
        """Stop the WebSocket client."""
        asyncio.create_task(self.disconnect())
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()


class MarketDataWebSocket(WebSocketClient):
    """WebSocket client for market data with built-in message parsing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.market_data_handlers: List[Callable[[MarketData], None]] = []
    
    async def _handle_message(self, message: str) -> None:
        """Handle incoming WebSocket message and parse market data."""
        try:
            data = json.loads(message)
            
            # Handle ping/pong
            if data.get('type') == 'ping':
                await self._send_pong()
                return
            elif data.get('type') == 'pong':
                self.last_pong = time.time()
                return
            
            # Parse market data
            market_data = self._parse_market_data(data)
            if market_data:
                for handler in self.market_data_handlers:
                    try:
                        handler(market_data)
                    except Exception as e:
                        logger.error(f"Error in market data handler: {str(e)}", exc_info=True)
            
            # Call parent handlers
            await super()._handle_message(message)
            
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON message: {message[:200]}...")
        except Exception as e:
            logger.error(f"Error processing market data: {str(e)}", exc_info=True)
    
    def _parse_market_data(self, data: Dict[str, Any]) -> Optional[MarketData]:
        """Parse market data from message (override in subclasses)."""
        try:
            # Example format: {'symbol': 'BTC/USD', 'bid': 50000.0, 'ask': 50001.0, 'bid_size': 1.0, 'ask_size': 0.5, 'timestamp': 1621234567.89}
            if 'symbol' in data and 'bid' in data and 'ask' in data:
                return MarketData(
                    symbol=data['symbol'],
                    timestamp=data.get('timestamp', time.time()),
                    bid=float(data['bid']),
                    ask=float(data['ask']),
                    bid_size=float(data.get('bid_size', 0)),
                    ask_size=float(data.get('ask_size', 0)),
                    exchange=data.get('exchange', 'unknown')
                )
            return None
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Error parsing market data: {str(e)}")
            return None
    
    def add_market_data_handler(self, handler: Callable[[MarketData], None]) -> None:
        """Add a market data handler."""
        if handler not in self.market_data_handlers:
            self.market_data_handlers.append(handler)
    
    def remove_market_data_handler(self, handler: Callable[[MarketData], None]) -> None:
        """Remove a market data handler."""
        if handler in self.market_data_handlers:
            self.market_data_handlers.remove(handler)


# Example usage
async def example_usage():
    """Example usage of the WebSocket client."""
    ws = MarketDataWebSocket("wss://api.example.com/ws")
    
    def on_market_data(data: MarketData):
        print(f"Market data: {data.symbol} bid={data.bid} ask={data.ask}")
    
    ws.add_market_data_handler(on_market_data)
    
    async with ws:
        await ws.subscribe(channel="ticker", symbols=["BTC/USD", "ETH/USD"])
        await asyncio.sleep(60)  # Run for 60 seconds

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    asyncio.run(example_usage())
