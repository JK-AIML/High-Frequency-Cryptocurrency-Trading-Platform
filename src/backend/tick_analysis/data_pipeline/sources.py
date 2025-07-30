"""
Data Source Module

This module provides data source implementations for various input methods
including WebSocket, Kafka, and file-based sources.
"""

import asyncio
import json
import logging
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, AsyncIterator, Set
import aiohttp
import aio_pika
import websockets
from kafka import KafkaConsumer
from kafka.errors import KafkaError
import pandas as pd

logger = logging.getLogger(__name__)

class DataSource(ABC):
    """Abstract base class for all data sources."""
    
    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the data source."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the data source."""
        pass
    
    @abstractmethod
    async def stream(self) -> AsyncIterator[Dict[str, Any]]:
        """Stream data from the source.
        
        Yields:
            Dictionary containing market data
        """
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if the source is connected."""
        pass

class WebSocketSource(DataSource):
    """WebSocket data source for real-time market data."""
    
    def __init__(self, 
                 url: str, 
                 subscriptions: List[Dict],
                 reconnect_interval: int = 5,
                 max_retries: int = 10):
        """
        Initialize WebSocket source.
        
        Args:
            url: WebSocket URL
            subscriptions: List of subscription messages
            reconnect_interval: Seconds between reconnection attempts
            max_retries: Maximum number of reconnection attempts
        """
        self.url = url
        self.subscriptions = subscriptions
        self.reconnect_interval = reconnect_interval
        self.max_retries = max_retries
        self._ws = None
        self._connected = False
        self._retry_count = 0
    
    async def connect(self) -> None:
        """Connect to WebSocket and subscribe to channels."""
        if self._connected:
            return
            
        while self._retry_count < self.max_retries:
            try:
                self._ws = await websockets.connect(self.url, ping_interval=None)
                logger.info(f"Connected to WebSocket: {self.url}")
                
                # Send subscription messages
                for sub in self.subscriptions:
                    await self._ws.send(json.dumps(sub))
                
                self._connected = True
                self._retry_count = 0
                return
                
            except Exception as e:
                self._retry_count += 1
                logger.error(f"WebSocket connection failed (attempt {self._retry_count}/{self.max_retries}): {e}")
                if self._retry_count >= self.max_retries:
                    raise
                await asyncio.sleep(self.reconnect_interval)
    
    async def disconnect(self) -> None:
        """Close WebSocket connection."""
        if self._ws:
            await self._ws.close()
            self._ws = None
        self._connected = False
    
    async def stream(self) -> AsyncIterator[Dict[str, Any]]:
        """Stream data from WebSocket."""
        if not self._connected:
            await self.connect()
            
        while self._connected:
            try:
                message = await self._ws.recv()
                data = json.loads(message)
                yield data
                
            except websockets.ConnectionClosed as e:
                logger.warning(f"WebSocket connection closed: {e}")
                await self._reconnect()
                
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")
                await asyncio.sleep(1)
    
    async def _reconnect(self) -> None:
        """Handle reconnection logic."""
        await self.disconnect()
        await asyncio.sleep(self.reconnect_interval)
        await self.connect()
    
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self._connected and self._ws is not None and not self._ws.closed

class KafkaSource(DataSource):
    """Kafka data source for distributed message streaming."""
    
    def __init__(self, 
                 bootstrap_servers: List[str],
                 topics: List[str],
                 group_id: str,
                 auto_offset_reset: str = 'latest',
                 **consumer_kwargs):
        """
        Initialize Kafka consumer.
        
        Args:
            bootstrap_servers: List of Kafka broker addresses
            topics: List of topics to subscribe to
            group_id: Consumer group ID
            auto_offset_reset: Where to start consuming from ('earliest' or 'latest')
            **consumer_kwargs: Additional Kafka consumer parameters
        """
        self.bootstrap_servers = bootstrap_servers
        self.topics = topics
        self.group_id = group_id
        self.auto_offset_reset = auto_offset_reset
        self.consumer_kwargs = consumer_kwargs
        self.consumer = None
        self._connected = False
    
    async def connect(self) -> None:
        """Connect to Kafka cluster."""
        if self._connected:
            return
            
        try:
            self.consumer = KafkaConsumer(
                *self.topics,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                auto_offset_reset=self.auto_offset_reset,
                enable_auto_commit=True,
                **self.consumer_kwargs
            )
            self._connected = True
            logger.info(f"Connected to Kafka: {self.bootstrap_servers}")
            
        except KafkaError as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Close Kafka consumer."""
        if self.consumer:
            self.consumer.close()
            self.consumer = None
        self._connected = False
    
    async def stream(self) -> AsyncIterator[Dict[str, Any]]:
        """Stream messages from Kafka."""
        if not self._connected:
            await self.connect()
            
        while self._connected:
            try:
                for message in self.consumer:
                    try:
                        data = json.loads(message.value.decode('utf-8'))
                        data['_kafka_metadata'] = {
                            'topic': message.topic,
                            'partition': message.partition,
                            'offset': message.offset,
                            'timestamp': message.timestamp
                        }
                        yield data
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to decode message: {message.value}")
                    except Exception as e:
                        logger.error(f"Error processing Kafka message: {e}")
                        
            except Exception as e:
                logger.error(f"Kafka consumer error: {e}")
                await asyncio.sleep(1)
                await self._reconnect()
    
    async def _reconnect(self) -> None:
        """Handle reconnection logic."""
        await self.disconnect()
        await asyncio.sleep(5)  # Wait before reconnecting
        await self.connect()
    
    def is_connected(self) -> bool:
        """Check if Kafka consumer is connected."""
        return self._connected and self.consumer is not None

class RabbitMQSource(DataSource):
    """RabbitMQ message queue source for distributed data streaming."""
    
    def __init__(self, 
                 url: str,
                 queue: str,
                 exchange: str = '',
                 exchange_type: str = 'direct',
                 routing_key: str = None,
                 durable: bool = True,
                 prefetch_count: int = 100,
                 reconnect_interval: int = 5):
        """
        Initialize RabbitMQ source.
        
        Args:
            url: RabbitMQ connection URL (amqp://user:pass@host:port/)
            queue: Queue name to consume from
            exchange: Exchange name (optional)
            exchange_type: Exchange type (direct, fanout, topic, headers)
            routing_key: Routing key for binding (optional)
            durable: Whether the queue should survive broker restarts
            prefetch_count: Maximum number of unacknowledged messages
            reconnect_interval: Seconds between reconnection attempts
        """
        self.url = url
        self.queue_name = queue
        self.exchange = exchange
        self.exchange_type = exchange_type
        self.routing_key = routing_key or queue
        self.durable = durable
        self.prefetch_count = prefetch_count
        self.reconnect_interval = reconnect_interval
        
        self._connection = None
        self._channel = None
        self._queue = None
        self._consumer_tag = None
        self._connected = False
        self._consuming = False
        self._message_queue = asyncio.Queue()
        self._consumer_task = None
    
    async def connect(self) -> None:
        """Establish connection to RabbitMQ."""
        if self._connected:
            return
            
        try:
            # Create connection
            self._connection = await aio_pika.connect_robust(
                self.url,
                client_properties={
                    'connection_name': f'tick-pipeline-{uuid.uuid4().hex[:8]}'
                }
            )
            
            # Create channel
            self._channel = await self._connection.channel()
            await self._channel.set_qos(prefetch_count=self.prefetch_count)
            
            # Declare exchange if specified
            if self.exchange:
                exchange = await self._channel.declare_exchange(
                    self.exchange,
                    aio_pika.ExchangeType[self.exchange_type.upper()],
                    durable=self.durable
                )
            else:
                exchange = None
            
            # Declare queue
            self._queue = await self._channel.declare_queue(
                self.queue_name,
                durable=self.durable,
                auto_delete=not self.durable
            )
            
            # Bind queue to exchange if exchange is specified
            if exchange:
                await self._queue.bind(exchange, routing_key=self.routing_key)
            
            self._connected = True
            logger.info(f"Connected to RabbitMQ: {self.url}")
            
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            await self.disconnect()
            raise
    
    async def disconnect(self) -> None:
        """Close RabbitMQ connection."""
        if self._consumer_task and not self._consumer_task.done():
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass
            
        if self._channel and not self._channel.is_closed:
            await self._channel.close()
            
        if self._connection and not self._connection.is_closed:
            await self._connection.close()
            
        self._connected = False
        self._consuming = False
    
    async def _on_message(self, message: aio_pika.IncomingMessage) -> None:
        """Process incoming message."""
        try:
            async with message.process():
                try:
                    # Parse message body as JSON
                    body = json.loads(message.body.decode())
                    # Add message metadata
                    body['_rabbitmq_metadata'] = {
                        'message_id': message.message_id,
                        'correlation_id': message.correlation_id,
                        'timestamp': message.timestamp,
                        'expiration': message.expiration,
                        'priority': message.priority,
                        'redelivered': message.redelivered,
                        'delivery_tag': message.delivery_tag,
                        'routing_key': message.routing_key,
                        'exchange': message.exchange,
                        'content_type': message.content_type,
                        'content_encoding': message.content_encoding,
                        'headers': dict(message.headers) if message.headers else {}
                    }
                    await self._message_queue.put(body)
                except json.JSONDecodeError:
                    logger.error(f"Failed to decode message: {message.body}")
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    async def _consume(self) -> None:
        """Start consuming messages."""
        while self._connected and not self._consuming:
            try:
                await self.connect()
                self._consumer_tag = await self._queue.consume(self._on_message)
                self._consuming = True
                logger.info(f"Started consuming from queue: {self.queue_name}")
                break
            except Exception as e:
                logger.error(f"Failed to start consumer: {e}")
                await asyncio.sleep(self.reconnect_interval)
    
    async def stream(self) -> AsyncIterator[Dict[str, Any]]:
        """Stream messages from RabbitMQ."""
        if not self._connected:
            await self.connect()
        
        # Start consumer task if not already running
        if not self._consuming and (self._consumer_task is None or self._consumer_task.done()):
            self._consumer_task = asyncio.create_task(self._consume())
        
        # Yield messages as they arrive
        while True:
            try:
                message = await self._message_queue.get()
                if message is None:  # Sentinel value for shutdown
                    break
                yield message
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in message stream: {e}")
                await asyncio.sleep(1)  # Prevent tight loop on errors
    
    async def ack(self, delivery_tag: int) -> None:
        """Acknowledge message processing."""
        if self._channel and not self._channel.is_closed:
            await self._channel.basic_ack(delivery_tag=delivery_tag)
    
    async def nack(self, delivery_tag: int, requeue: bool = True) -> None:
        """Negatively acknowledge message processing."""
        if self._channel and not self._channel.is_closed:
            await self._channel.basic_nack(delivery_tag=delivery_tag, requeue=requeue)
    
    def is_connected(self) -> bool:
        """Check if connected to RabbitMQ."""
        return self._connected and self._consuming


class FileSource(DataSource):
    """File-based data source for backtesting and development."""
    
    def __init__(self, 
                 file_path: str, 
                 file_format: str = 'parquet',
                 chunk_size: int = 1000):
        """
        Initialize file source.
        
        Args:
            file_path: Path to the data file
            file_format: File format ('parquet', 'csv', 'json')
            chunk_size: Number of rows to process at a time
        """
        self.file_path = file_path
        self.file_format = file_format.lower()
        self.chunk_size = chunk_size
        self._file = None
        self._connected = False
        self._reader = None
    
    async def connect(self) -> None:
        """Open the data file."""
        if self._connected:
            return
            
        try:
            if self.file_format == 'parquet':
                import pyarrow.parquet as pq
                self._file = pq.ParquetFile(self.file_path)
                self._reader = self._file.iter_batches(batch_size=self.chunk_size)
            elif self.file_format == 'csv':
                self._reader = pd.read_csv(
                    self.file_path, 
                    chunksize=self.chunk_size,
                    iterator=True
                )
            elif self.file_format == 'json':
                self._file = open(self.file_path, 'r')
                self._reader = pd.read_json(
                    self._file, 
                    lines=True,
                    chunksize=self.chunk_size
                )
            else:
                raise ValueError(f"Unsupported file format: {self.file_format}")
                
            self._connected = True
            logger.info(f"Opened file: {self.file_path}")
            
        except Exception as e:
            logger.error(f"Failed to open file {self.file_path}: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Close the data file."""
        if self._file and not self._file.closed:
            self._file.close()
        self._file = None
        self._reader = None
        self._connected = False
    
    async def stream(self) -> AsyncIterator[Dict[str, Any]]:
        """Stream data from file."""
        if not self._connected:
            await self.connect()
            
        try:
            for chunk in self._reader:
                if self.file_format == 'parquet':
                    df = chunk.to_pandas()
                else:
                    df = chunk
                    
                for _, row in df.iterrows():
                    yield row.to_dict()
                    
        except StopIteration:
            logger.info("Reached end of file")
            
        except Exception as e:
            logger.error(f"Error reading from file: {e}")
            raise
    
    def is_connected(self) -> bool:
        """Check if file is open."""
        return self._connected and self._file is not None and not self._file.closed
