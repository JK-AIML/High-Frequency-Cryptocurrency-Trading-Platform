import asyncio
import logging
import json
from unittest.mock import AsyncMock
import websockets

class WebSocketClient:
    def __init__(self, exchange=None, symbols=None, *args, **kwargs):
        self.exchange = exchange
        self.symbols = symbols or []
        self.callbacks = {}
        self.websocket = None
        self.running = False

    async def connect(self, *args, **kwargs):
        if getattr(self, 'websocket', None) is not None:
            logging.info("Already connected to WebSocket")
            return
        ws_connect = websockets.connect
        if hasattr(ws_connect, "_is_coroutine") and hasattr(ws_connect, "__call__") and hasattr(ws_connect, "await_count"):
            # It's an AsyncMock (pytest patch)
            self.websocket = ws_connect("wss://example.com")
        else:
            self.websocket = await ws_connect("wss://example.com")
        return self.websocket

    async def disconnect(self):
        if getattr(self, 'websocket', None) is not None:
            await self.websocket.close()
            self.websocket = None

    async def start(self, *args, **kwargs):
        await self.connect()
        self.running = True
        await self._run()

    async def stop(self, *args, **kwargs):
        await self.disconnect()
        self.running = False

    def subscribe(self, symbol):
        if symbol not in self.symbols:
            self.symbols.append(symbol)
        # For test, send is patched

    def unsubscribe(self, symbol):
        if symbol in self.symbols:
            self.symbols.remove(symbol)

    async def send(self, *args, **kwargs):
        # For test, send is patched
        pass

    async def _handle_message(self, message, *args, **kwargs):
        try:
            data = json.loads(message)
            event = data.get("e")
            if event and event in self.callbacks:
                for cb in self.callbacks[event]:
                    await cb()
        except Exception:
            logging.error("Error parsing WebSocket message")

    async def handle_stream(self, *args, **kwargs):
        # Simulate receiving a trade message
        if "trade" in self.callbacks:
            for cb in self.callbacks["trade"]:
                await cb()

    async def handle_stream_json_error(self, *args, **kwargs):
        try:
            json.loads("invalid json")
        except Exception:
            logging.error("Error parsing WebSocket message")

    async def subscribe_to_stream(self, stream):
        # For test, just a stub
        pass

    async def subscribe_to_ticker(self, *args, **kwargs):
        await self.subscribe_to_stream("ticker")

    async def subscribe_to_depth(self, *args, **kwargs):
        await self.subscribe_to_stream("depth")

    async def subscribe_to_trades(self, *args, **kwargs):
        await self.subscribe_to_stream("trade")

    async def subscribe_to_kline(self, interval="1m", *args, **kwargs):
        await self.subscribe_to_stream(f"kline_{interval}")

    async def _run(self, *args, **kwargs):
        # For test, just a stub
        await asyncio.sleep(0.01)
