from fastapi import WebSocket, WebSocketDisconnect
from typing import List, Dict, Any
import json
import asyncio
import logging
from datetime import datetime
import websockets
import aiohttp
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.market_data_clients: Dict[str, Any] = {}
        # New: track last sent portfolio, trade, and risk data for efficient updates
        self.last_portfolio: Dict = {}
        self.last_trade: Dict = {}
        self.last_risk: Dict = {}
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"New WebSocket connection. Total connections: {len(self.active_connections)}")
        # On connect, send latest state if available
        if self.last_portfolio:
            await self.send_personal_message(json.dumps({"type": "portfolioUpdate", "data": self.last_portfolio}), websocket)
        if self.last_trade:
            await self.send_personal_message(json.dumps({"type": "tradeUpdate", "data": self.last_trade}), websocket)
        if self.last_risk:
            await self.send_personal_message(json.dumps({"type": "riskUpdate", "data": self.last_risk}), websocket)
        
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"WebSocket disconnected. Remaining connections: {len(self.active_connections)}")
        
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except WebSocketDisconnect:
                self.disconnect(connection)
                
    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except WebSocketDisconnect:
            self.disconnect(websocket)

    # New: broadcast portfolio update
    async def broadcast_portfolio(self, portfolio_data: Dict):
        self.last_portfolio = portfolio_data
        message = json.dumps({"type": "portfolioUpdate", "data": portfolio_data})
        await self.broadcast(message)

    # New: broadcast trade update
    async def broadcast_trade(self, trade_data: Dict):
        self.last_trade = trade_data
        message = json.dumps({"type": "tradeUpdate", "data": trade_data})
        await self.broadcast(message)

    # New: broadcast risk update
    async def broadcast_risk(self, risk_data: Dict):
        self.last_risk = risk_data
        message = json.dumps({"type": "riskUpdate", "data": risk_data})
        await self.broadcast(message)

    # New: generic update (for future extensibility)
    async def broadcast_update(self, update_type: str, data: Dict):
        message = json.dumps({"type": update_type, "data": data})
        await self.broadcast(message)

class MarketDataClient:
    def __init__(self):
        self.binance_ws_url = "wss://stream.binance.com:9443/ws"
        self.polygon_ws_url = "wss://socket.polygon.io/stocks"
        self.cryptocompare_ws_url = "wss://streamer.cryptocompare.com/v2"
        self.api_keys = {
            "polygon": os.getenv("POLYGON_API_KEY"),
            "cryptocompare": os.getenv("CRYPTOCOMPARE_API_KEY")
        }
        
    async def connect_binance(self, symbols: List[str]):
        """Connect to Binance WebSocket API"""
        try:
            async with websockets.connect(self.binance_ws_url) as websocket:
                # Subscribe to ticker streams
                subscribe_message = {
                    "method": "SUBSCRIBE",
                    "params": [f"{symbol.lower()}@ticker" for symbol in symbols],
                    "id": 1
                }
                await websocket.send(json.dumps(subscribe_message))
                
                while True:
                    try:
                        message = await websocket.recv()
                        data = json.loads(message)
                        # Process and format the data
                        processed_data = self._process_binance_data(data)
                        yield processed_data
                    except websockets.exceptions.ConnectionClosed:
                        logger.error("Binance WebSocket connection closed")
                        break
        except Exception as e:
            logger.error(f"Error in Binance WebSocket connection: {str(e)}")
            
    async def connect_polygon(self, symbols: List[str]):
        """Connect to Polygon.io WebSocket API"""
        try:
            async with websockets.connect(self.polygon_ws_url) as websocket:
                # Authenticate
                auth_message = {
                    "action": "auth",
                    "params": self.api_keys["polygon"]
                }
                await websocket.send(json.dumps(auth_message))
                
                # Subscribe to trades
                subscribe_message = {
                    "action": "subscribe",
                    "params": [f"T.{symbol}" for symbol in symbols]
                }
                await websocket.send(json.dumps(subscribe_message))
                
                while True:
                    try:
                        message = await websocket.recv()
                        data = json.loads(message)
                        # Process and format the data
                        processed_data = self._process_polygon_data(data)
                        yield processed_data
                    except websockets.exceptions.ConnectionClosed:
                        logger.error("Polygon WebSocket connection closed")
                        break
        except Exception as e:
            logger.error(f"Error in Polygon WebSocket connection: {str(e)}")
            
    def _process_binance_data(self, data: Dict) -> Dict:
        """Process and format Binance WebSocket data"""
        try:
            return {
                "source": "binance",
                "symbol": data.get("s"),
                "price": float(data.get("c", 0)),
                "volume": float(data.get("v", 0)),
                "timestamp": datetime.utcnow().isoformat(),
                "raw_data": data
            }
        except Exception as e:
            logger.error(f"Error processing Binance data: {str(e)}")
            return {}
            
    def _process_polygon_data(self, data: Dict) -> Dict:
        """Process and format Polygon.io WebSocket data"""
        try:
            return {
                "source": "polygon",
                "symbol": data.get("sym"),
                "price": float(data.get("p", 0)),
                "volume": float(data.get("s", 0)),
                "timestamp": datetime.utcnow().isoformat(),
                "raw_data": data
            }
        except Exception as e:
            logger.error(f"Error processing Polygon data: {str(e)}")
            return {}

# Initialize connection manager
manager = ConnectionManager()
market_data_client = MarketDataClient()

async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                request = json.loads(data)
                action = request.get("action")
                symbols = request.get("symbols", [])
                
                if action == "subscribe":
                    # Start market data streams
                    asyncio.create_task(handle_market_data(websocket, symbols))
                elif action == "unsubscribe":
                    # Handle unsubscribe logic
                    pass
                    
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "error": "Invalid JSON format"
                }))
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        manager.disconnect(websocket)

async def handle_market_data(websocket: WebSocket, symbols: List[str]):
    """Handle market data streams for a specific WebSocket connection"""
    try:
        # Start Binance stream
        async for binance_data in market_data_client.connect_binance(symbols):
            if binance_data:
                await manager.send_personal_message(
                    json.dumps(binance_data),
                    websocket
                )
                
        # Start Polygon stream
        async for polygon_data in market_data_client.connect_polygon(symbols):
            if polygon_data:
                await manager.send_personal_message(
                    json.dumps(polygon_data),
                    websocket
                )
                
    except Exception as e:
        logger.error(f"Error in market data handling: {str(e)}")
        await manager.send_personal_message(
            json.dumps({"error": "Market data stream error"}),
            websocket
        ) 