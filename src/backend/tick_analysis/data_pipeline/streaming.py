"""
Real-time Data Streaming Module

This module provides real-time data streaming capabilities including
WebSocket integration and message queue support for data ingestion.
"""

import logging
import asyncio
import json
from typing import Dict, List, Any, Optional, Union, Callable, Awaitable
from dataclasses import dataclass
from datetime import datetime
import websockets
import aiohttp
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
import aiormq
from aiormq.abc import AbstractChannel, AbstractConnection
import backoff

logger = logging.getLogger(__name__)

@dataclass
class WebSocketConfig:
    """Configuration for WebSocket connection."""
    url: str
    protocols: Optional[List[str]] = None
    headers: Optional[Dict[str, str]] = None
    ping_interval: int = 30
    ping_timeout: int = 10
    close_timeout: int = 10
    max_size: int = 2**20  # 1MB
    max_queue: int = 2**10  # 1024
    compression: Optional[str] = None

@dataclass
class KafkaConfig:
    """Configuration for Kafka connection."""
    bootstrap_servers: List[str]
    topic: str
    group_id: Optional[str] = None
    client_id: Optional[str] = None
    security_protocol: str = "PLAINTEXT"
    sasl_mechanism: Optional[str] = None
    sasl_plain_username: Optional[str] = None
    sasl_plain_password: Optional[str] = None
    ssl_cafile: Optional[str] = None
    auto_offset_reset: str = "latest"
    enable_auto_commit: bool = True
    auto_commit_interval_ms: int = 5000
    max_poll_interval_ms: int = 300000
    session_timeout_ms: int = 10000
    heartbeat_interval_ms: int = 3000

@dataclass
class RabbitMQConfig:
    """Configuration for RabbitMQ connection."""
    host: str
    port: int = 5672
    username: str = "guest"
    password: str = "guest"
    virtual_host: str = "/"
    exchange: str = "tick_data"
    exchange_type: str = "topic"
    queue: str = "tick_data_queue"
    routing_key: str = "tick.data.#"
    prefetch_count: int = 100
    heartbeat: int = 60
    ssl: bool = False
    ssl_options: Optional[Dict[str, Any]] = None

class WebSocketStream:
    """WebSocket-based data streaming."""
    
    def __init__(self, config: WebSocketConfig):
        """
        Initialize WebSocket stream.
        
        Args:
            config: WebSocket configuration
        """
        self.config = config
        self.ws = None
        self._connected = False
        self._message_queue = asyncio.Queue(maxsize=config.max_queue)
        self._handlers = []
    
    @backoff.on_exception(backoff.expo, Exception, max_tries=5)
    async def connect(self) -> None:
        """Establish WebSocket connection."""
        if self._connected:
            return
            
        try:
            self.ws = await websockets.connect(
                self.config.url,
                subprotocols=self.config.protocols,
                extra_headers=self.config.headers,
                ping_interval=self.config.ping_interval,
                ping_timeout=self.config.ping_timeout,
                close_timeout=self.config.close_timeout,
                max_size=self.config.max_size,
                compression=self.config.compression
            )
            self._connected = True
            logger.info(f"Connected to WebSocket at {self.config.url}")
            
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
            self._connected = False
            raise
    
    async def subscribe(self, channels: List[str]) -> None:
        """
        Subscribe to WebSocket channels.
        
        Args:
            channels: List of channel names to subscribe to
        """
        if not self._connected:
            await self.connect()
            
        try:
            for channel in channels:
                await self.ws.send(json.dumps({
                    "type": "subscribe",
                    "channel": channel
                }))
            logger.info(f"Subscribed to channels: {channels}")
            
        except Exception as e:
            logger.error(f"Failed to subscribe to channels: {e}")
            self._connected = False
            raise
    
    async def start(self) -> None:
        """Start receiving messages."""
        if not self._connected:
            await self.connect()
            
        try:
            while True:
                try:
                    message = await self.ws.recv()
                    data = json.loads(message)
                    
                    # Put message in queue
                    await self._message_queue.put(data)
                    
                    # Call handlers
                    for handler in self._handlers:
                        try:
                            await handler(data)
                        except Exception as e:
                            logger.error(f"Error in message handler: {e}")
                            
                except websockets.exceptions.ConnectionClosed:
                    logger.warning("WebSocket connection closed, attempting to reconnect...")
                    self._connected = False
                    await self.connect()
                    
        except Exception as e:
            logger.error(f"Error in WebSocket stream: {e}")
            self._connected = False
            raise
    
    def add_handler(self, handler: Callable[[Dict[str, Any]], Awaitable[None]]) -> None:
        """
        Add message handler.
        
        Args:
            handler: Async function to handle messages
        """
        self._handlers.append(handler)
    
    async def get_message(self) -> Dict[str, Any]:
        """
        Get next message from queue.
        
        Returns:
            Next message
        """
        return await self._message_queue.get()
    
    async def close(self) -> None:
        """Close WebSocket connection."""
        if self.ws:
            await self.ws.close()
        self._connected = False

class KafkaStream:
    """Kafka-based data streaming."""
    
    def __init__(self, config: KafkaConfig):
        """
        Initialize Kafka stream.
        
        Args:
            config: Kafka configuration
        """
        self.config = config
        self.producer = None
        self.consumer = None
        self._connected = False
        self._handlers = []
    
    @backoff.on_exception(backoff.expo, Exception, max_tries=5)
    async def connect(self) -> None:
        """Establish Kafka connection."""
        if self._connected:
            return
            
        try:
            # Create producer
            self.producer = AIOKafkaProducer(
                bootstrap_servers=self.config.bootstrap_servers,
                client_id=self.config.client_id,
                security_protocol=self.config.security_protocol,
                sasl_mechanism=self.config.sasl_mechanism,
                sasl_plain_username=self.config.sasl_plain_username,
                sasl_plain_password=self.config.sasl_plain_password,
                ssl_cafile=self.config.ssl_cafile
            )
            await self.producer.start()
            
            # Create consumer
            self.consumer = AIOKafkaConsumer(
                self.config.topic,
                bootstrap_servers=self.config.bootstrap_servers,
                group_id=self.config.group_id,
                client_id=self.config.client_id,
                security_protocol=self.config.security_protocol,
                sasl_mechanism=self.config.sasl_mechanism,
                sasl_plain_username=self.config.sasl_plain_username,
                sasl_plain_password=self.config.sasl_plain_password,
                ssl_cafile=self.config.ssl_cafile,
                auto_offset_reset=self.config.auto_offset_reset,
                enable_auto_commit=self.config.enable_auto_commit,
                auto_commit_interval_ms=self.config.auto_commit_interval_ms,
                max_poll_interval_ms=self.config.max_poll_interval_ms,
                session_timeout_ms=self.config.session_timeout_ms,
                heartbeat_interval_ms=self.config.heartbeat_interval_ms
            )
            await self.consumer.start()
            
            self._connected = True
            logger.info(f"Connected to Kafka at {self.config.bootstrap_servers}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            self._connected = False
            raise
    
    async def produce(self, data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> None:
        """
        Produce messages to Kafka.
        
        Args:
            data: Single message or list of messages
        """
        if not self._connected:
            await self.connect()
            
        if not isinstance(data, list):
            data = [data]
            
        try:
            for message in data:
                await self.producer.send_and_wait(
                    self.config.topic,
                    json.dumps(message).encode()
                )
                
        except Exception as e:
            logger.error(f"Failed to produce message to Kafka: {e}")
            self._connected = False
            raise
    
    async def start(self) -> None:
        """Start consuming messages."""
        if not self._connected:
            await self.connect()
            
        try:
            async for message in self.consumer:
                try:
                    data = json.loads(message.value.decode())
                    
                    # Call handlers
                    for handler in self._handlers:
                        try:
                            await handler(data)
                        except Exception as e:
                            logger.error(f"Error in message handler: {e}")
                            
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    
        except Exception as e:
            logger.error(f"Error in Kafka stream: {e}")
            self._connected = False
            raise
    
    def add_handler(self, handler: Callable[[Dict[str, Any]], Awaitable[None]]) -> None:
        """
        Add message handler.
        
        Args:
            handler: Async function to handle messages
        """
        self._handlers.append(handler)
    
    async def close(self) -> None:
        """Close Kafka connection."""
        if self.producer:
            await self.producer.stop()
        if self.consumer:
            await self.consumer.stop()
        self._connected = False

class RabbitMQStream:
    """RabbitMQ-based data streaming."""
    
    def __init__(self, config: RabbitMQConfig):
        """
        Initialize RabbitMQ stream.
        
        Args:
            config: RabbitMQ configuration
        """
        self.config = config
        self.connection = None
        self.channel = None
        self._connected = False
        self._handlers = []
    
    @backoff.on_exception(backoff.expo, Exception, max_tries=5)
    async def connect(self) -> None:
        """Establish RabbitMQ connection."""
        if self._connected:
            return
            
        try:
            # Create connection
            self.connection = await aiormq.connect(
                host=self.config.host,
                port=self.config.port,
                login=self.config.username,
                password=self.config.password,
                virtualhost=self.config.virtual_host,
                heartbeat=self.config.heartbeat,
                ssl=self.config.ssl,
                ssl_options=self.config.ssl_options
            )
            
            # Create channel
            self.channel = await self.connection.channel()
            await self.channel.set_qos(prefetch_count=self.config.prefetch_count)
            
            # Declare exchange
            await self.channel.exchange_declare(
                exchange=self.config.exchange,
                exchange_type=self.config.exchange_type,
                durable=True
            )
            
            # Declare queue
            await self.channel.queue_declare(
                queue=self.config.queue,
                durable=True
            )
            
            # Bind queue to exchange
            await self.channel.queue_bind(
                queue=self.config.queue,
                exchange=self.config.exchange,
                routing_key=self.config.routing_key
            )
            
            self._connected = True
            logger.info(f"Connected to RabbitMQ at {self.config.host}:{self.config.port}")
            
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            self._connected = False
            raise
    
    async def produce(self, data: Union[Dict[str, Any], List[Dict[str, Any]]], routing_key: Optional[str] = None) -> None:
        """
        Produce messages to RabbitMQ.
        
        Args:
            data: Single message or list of messages
            routing_key: Optional routing key
        """
        if not self._connected:
            await self.connect()
            
        if not isinstance(data, list):
            data = [data]
            
        try:
            for message in data:
                await self.channel.basic_publish(
                    json.dumps(message).encode(),
                    exchange=self.config.exchange,
                    routing_key=routing_key or self.config.routing_key
                )
                
        except Exception as e:
            logger.error(f"Failed to produce message to RabbitMQ: {e}")
            self._connected = False
            raise
    
    async def start(self) -> None:
        """Start consuming messages."""
        if not self._connected:
            await self.connect()
            
        try:
            # Declare consumer
            await self.channel.basic_consume(
                self.config.queue,
                self._process_message
            )
            
            # Keep connection alive
            while True:
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Error in RabbitMQ stream: {e}")
            self._connected = False
            raise
    
    async def _process_message(self, message: aiormq.abc.DeliveredMessage) -> None:
        """
        Process received message.
        
        Args:
            message: Received message
        """
        try:
            data = json.loads(message.body.decode())
            
            # Call handlers
            for handler in self._handlers:
                try:
                    await handler(data)
                except Exception as e:
                    logger.error(f"Error in message handler: {e}")
                    
            # Acknowledge message
            await self.channel.basic_ack(message.delivery_tag)
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            # Reject message
            await self.channel.basic_nack(message.delivery_tag)
    
    def add_handler(self, handler: Callable[[Dict[str, Any]], Awaitable[None]]) -> None:
        """
        Add message handler.
        
        Args:
            handler: Async function to handle messages
        """
        self._handlers.append(handler)
    
    async def close(self) -> None:
        """Close RabbitMQ connection."""
        if self.channel:
            await self.channel.close()
        if self.connection:
            await self.connection.close()
        self._connected = False 