"""
Message queue integration for reliable message handling.
"""
import json
import logging
import asyncio
from typing import Dict, List, Optional, Callable, Any, Union
import aio_pika
from aio_pika import connect_robust, ExchangeType, Message, DeliveryMode
from aio_pika.abc import AbstractIncomingMessage, AbstractChannel, AbstractExchange, AbstractQueue
from dataclasses import asdict, is_dataclass
import time
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)

class MessageQueue:
    """
    Message queue client for reliable message handling.
    
    Features:
    - Publish/subscribe pattern
    - Message persistence
    - Automatic reconnection
    - Message acknowledgment
    - Dead letter queue for failed messages
    """
    
    def __init__(
        self,
        url: str = "amqp://guest:guest@localhost/",
        exchange_name: str = "tick_data",
        queue_name: str = "tick_processor",
        exchange_type: ExchangeType = ExchangeType.TOPIC,
        max_retries: int = 5,
        retry_delay: float = 1.0,
        prefetch_count: int = 100,
        **kwargs
    ):
        """
        Initialize message queue client.
        
        Args:
            url: RabbitMQ connection URL
            exchange_name: Name of the exchange
            queue_name: Name of the queue
            exchange_type: Type of exchange (direct, topic, fanout, headers)
            max_retries: Maximum number of connection retries
            retry_delay: Delay between retries in seconds
            prefetch_count: Maximum number of unacknowledged messages
            **kwargs: Additional connection parameters
        """
        self.url = url
        self.exchange_name = exchange_name
        self.queue_name = queue_name
        self.exchange_type = exchange_type
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.prefetch_count = prefetch_count
        self.connection_params = kwargs
        
        self.connection: Optional[aio_pika.RobustConnection] = None
        self.channel: Optional[AbstractChannel] = None
        self.exchange: Optional[AbstractExchange] = None
        self.queue: Optional[AbstractQueue] = None
        self.consumer_tag: Optional[str] = None
        self.running = False
        self._message_handlers: List[Callable[[Dict[str, Any], str], None]] = []
        self._error_handlers: List[Callable[[Exception, Dict[str, Any]], None]] = []
        self._reconnect_task: Optional[asyncio.Task] = None
    
    async def connect(self) -> None:
        """Connect to the message broker and set up the exchange and queue."""
        if self.connection and not self.connection.is_closed:
            return
            
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Connecting to message broker (attempt {attempt + 1}/{self.max_retries})...")
                self.connection = await connect_robust(self.url, **self.connection_params)
                
                # Set up connection closed callback
                self.connection.add_close_callback(self._on_connection_closed)
                
                # Create channel
                self.channel = await self.connection.channel()
                await self.channel.set_qos(prefetch_count=self.prefetch_count)
                
                # Declare exchange
                self.exchange = await self.channel.declare_exchange(
                    self.exchange_name,
                    self.exchange_type,
                    durable=True,
                    auto_delete=False
                )
                
                # Declare queue with dead letter exchange
                args = {
                    'x-dead-letter-exchange': f"{self.exchange_name}.dlx",
                    'x-dead-letter-routing-key': f"{self.queue_name}.dlq"
                }
                
                self.queue = await self.channel.declare_queue(
                    self.queue_name,
                    durable=True,
                    arguments=args
                )
                
                # Bind queue to exchange with routing key
                await self.queue.bind(self.exchange, routing_key=f"{self.queue_name}.#")
                
                # Set up dead letter exchange and queue
                await self._setup_dead_letter_queue()
                
                logger.info("Successfully connected to message broker")
                return
                
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error("Failed to connect to message broker after multiple attempts", exc_info=True)
                    raise
                    
                logger.warning(
                    f"Connection attempt {attempt + 1} failed: {str(e)}. "
                    f"Retrying in {self.retry_delay} seconds..."
                )
                await asyncio.sleep(self.retry_delay)
    
    async def _setup_dead_letter_queue(self) -> None:
        """Set up dead letter exchange and queue."""
        if not self.channel:
            return
            
        # Declare dead letter exchange
        dlx_exchange = await self.channel.declare_exchange(
            f"{self.exchange_name}.dlx",
            ExchangeType.TOPIC,
            durable=True
        )
        
        # Declare dead letter queue
        dlq = await self.channel.declare_queue(
            f"{self.queue_name}.dlq",
            durable=True
        )
        
        # Bind dead letter queue to exchange
        await dlq.bind(dlx_exchange, routing_key=f"{self.queue_name}.dlq")
    
    def _on_connection_closed(self, connection, exception=None):
        """Handle connection closed event."""
        logger.warning("Connection to message broker closed")
        if exception:
            logger.error(f"Connection closed due to: {str(exception)}")
        
        # Schedule reconnection if not already reconnecting
        if not self._reconnect_task or self._reconnect_task.done():
            self._reconnect_task = asyncio.create_task(self._reconnect())
    
    async def _reconnect(self) -> None:
        """Handle reconnection logic."""
        if self.connection and not self.connection.is_closed:
            return
            
        logger.info("Attempting to reconnect to message broker...")
        
        while True:
            try:
                await self.connect()
                
                # Resubscribe if we were consuming
                if self.running and self._message_handlers:
                    await self.start_consuming()
                    
                logger.info("Successfully reconnected to message broker")
                return
                
            except Exception as e:
                logger.error(f"Reconnection failed: {str(e)}")
                await asyncio.sleep(min(self.retry_delay * 2, 30))  # Exponential backoff with max 30s
    
    async def publish(
        self,
        message: Union[Dict, str, bytes],
        routing_key: str = "",
        persistent: bool = True,
        headers: Optional[Dict[str, Any]] = None,
        message_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        expiration: Optional[Union[int, float, str]] = None,
        priority: Optional[int] = None
    ) -> None:
        """
        Publish a message to the exchange.
        
        Args:
            message: Message to publish (dict, str, or bytes)
            routing_key: Routing key for the message
            persistent: Whether to make the message persistent
            headers: Optional message headers
            message_id: Optional message ID
            correlation_id: Optional correlation ID
            expiration: Optional message expiration in seconds or as a string (e.g., '60000' for 60 seconds)
            priority: Optional message priority (0-9)
        """
        if not self.exchange:
            raise RuntimeError("Not connected to message broker")
            
        try:
            # Convert message to bytes if needed
            if isinstance(message, dict) or is_dataclass(message):
                message_data = json.dumps(asdict(message) if is_dataclass(message) else message).encode('utf-8')
            elif isinstance(message, str):
                message_data = message.encode('utf-8')
            elif isinstance(message, bytes):
                message_data = message
            else:
                raise ValueError(f"Unsupported message type: {type(message)}")
            
            # Create message properties
            delivery_mode = DeliveryMode.PERSISTENT if persistent else DeliveryMode.NOT_PERSISTENT
            
            properties = {
                'delivery_mode': delivery_mode,
                'timestamp': int(time.time()),
                'message_id': message_id or str(uuid.uuid4()),
                'headers': headers or {},
            }
            
            if correlation_id:
                properties['correlation_id'] = correlation_id
                
            if expiration is not None:
                properties['expiration'] = str(int(expiration * 1000)) if isinstance(expiration, (int, float)) else str(expiration)
                
            if priority is not None:
                properties['priority'] = min(max(0, priority), 9)
            
            # Publish message
            await self.exchange.publish(
                Message(body=message_data, **properties),
                routing_key=routing_key or self.queue_name
            )
            
            logger.debug(f"Published message to {routing_key or self.queue_name}")
            
        except Exception as e:
            logger.error(f"Failed to publish message: {str(e)}", exc_info=True)
            raise
    
    async def _process_message(self, message: AbstractIncomingMessage) -> None:
        """Process an incoming message."""
        try:
            # Parse message body
            try:
                body = message.body.decode('utf-8')
                try:
                    data = json.loads(body)
                except json.JSONDecodeError:
                    data = {'raw': body}
            except UnicodeDecodeError:
                data = {'raw': message.body.hex()}
            
            # Add metadata
            metadata = {
                'message_id': message.message_id,
                'correlation_id': message.correlation_id,
                'timestamp': message.timestamp or time.time(),
                'headers': dict(message.headers) if message.headers else {},
                'routing_key': message.routing_key or '',
                'exchange': message.exchange or '',
                'redelivered': message.redelivered,
                'delivery_tag': message.delivery_tag
            }
            
            # Call message handlers
            handled = False
            for handler in self._message_handlers:
                try:
                    handler(data, metadata)
                    handled = True
                except Exception as e:
                    logger.error(f"Error in message handler: {str(e)}", exc_info=True)
            
            # Acknowledge message if at least one handler processed it successfully
            if handled:
                await message.ack()
                logger.debug(f"Processed message: {message.message_id}")
            else:
                await message.nack(requeue=False)  # Send to DLQ
                logger.warning(f"No handler processed message: {message.message_id}")
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}", exc_info=True)
            
            # Call error handlers
            for handler in self._error_handlers:
                try:
                    handler(e, getattr(message, 'body', None))
                except Exception as handler_error:
                    logger.error(f"Error in error handler: {str(handler_error)}", exc_info=True)
            
            # Nack the message (don't requeue, let it go to DLQ)
            if not message.processed:
                await message.nack(requeue=False)
    
    async def start_consuming(self) -> None:
        """Start consuming messages from the queue."""
        if not self.queue:
            raise RuntimeError("Queue not declared. Call connect() first.")
            
        if self.running:
            logger.warning("Already consuming messages")
            return
            
        self.running = True
        
        try:
            # Start consuming
            await self.queue.consume(self._process_message, no_ack=False)
            logger.info(f"Started consuming messages from {self.queue_name}")
            
            # Keep the connection alive
            while self.running:
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Error in consumer: {str(e)}", exc_info=True)
            self.running = False
            raise
    
    async def stop_consuming(self) -> None:
        """Stop consuming messages."""
        self.running = False
        
        if self.consumer_tag and self.channel:
            try:
                await self.channel.basic_cancel(self.consumer_tag)
                logger.info("Stopped consuming messages")
            except Exception as e:
                logger.error(f"Error stopping consumer: {str(e)}")
    
    def add_message_handler(self, handler: Callable[[Dict, Dict], None]) -> None:
        """Add a message handler function."""
        if handler not in self._message_handlers:
            self._message_handlers.append(handler)
            logger.debug(f"Added message handler: {handler.__name__}")
    
    def remove_message_handler(self, handler: Callable[[Dict, Dict], None]) -> None:
        """Remove a message handler function."""
        if handler in self._message_handlers:
            self._message_handlers.remove(handler)
            logger.debug(f"Removed message handler: {handler.__name__}")
    
    def add_error_handler(self, handler: Callable[[Exception, Any], None]) -> None:
        """Add an error handler function."""
        if handler not in self._error_handlers:
            self._error_handlers.append(handler)
            logger.debug(f"Added error handler: {handler.__name__}")
    
    def remove_error_handler(self, handler: Callable[[Exception, Any], None]) -> None:
        """Remove an error handler function."""
        if handler in self._error_handlers:
            self._error_handlers.remove(handler)
            logger.debug(f"Removed error handler: {handler.__name__}")
    
    async def close(self) -> None:
        """Close the connection to the message broker."""
        await self.stop_consuming()
        
        if self.connection and not self.connection.is_closed:
            await self.connection.close()
            logger.info("Closed connection to message broker")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


class MarketDataQueue(MessageQueue):
    """Specialized message queue for market data with common patterns."""
    
    def __init__(
        self,
        symbol: str,
        data_type: str = "tick",  # tick, ohlcv, orderbook, etc.
        **kwargs
    ):
        """
        Initialize market data queue.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USD')
            data_type: Type of market data
            **kwargs: Additional MessageQueue parameters
        """
        queue_name = kwargs.pop('queue_name', f"market_data.{symbol.lower().replace('/', '_')}.{data_type}")
        exchange_name = kwargs.pop('exchange_name', 'market_data')
        
        super().__init__(
            exchange_name=exchange_name,
            queue_name=queue_name,
            **kwargs
        )
        
        self.symbol = symbol
        self.data_type = data_type
        self._market_data_handlers: List[Callable[[Dict], None]] = []
    
    async def publish_market_data(self, data: Dict) -> None:
        """Publish market data to the queue."""
        if 'symbol' not in data:
            data['symbol'] = self.symbol
            
        if 'timestamp' not in data:
            data['timestamp'] = time.time()
            
        routing_key = f"{self.symbol}.{self.data_type}"
        
        await self.publish(
            message=data,
            routing_key=routing_key,
            persistent=True,
            headers={
                'symbol': self.symbol,
                'data_type': self.data_type,
                'timestamp': str(time.time())
            }
        )
    
    def add_market_data_handler(self, handler: Callable[[Dict], None]) -> None:
        """Add a market data handler."""
        if handler not in self._market_data_handlers:
            self._market_data_handlers.append(handler)
            
            # Wrap the handler to extract the message body
            async def wrapped_handler(message: Dict, metadata: Dict) -> None:
                try:
                    handler(message)
                except Exception as e:
                    logger.error(f"Error in market data handler: {str(e)}", exc_info=True)
            
            self.add_message_handler(wrapped_handler)
    
    async def start_consuming(self) -> None:
        """Start consuming market data messages."""
        if not self._market_data_handlers:
            logger.warning("No market data handlers registered")
            return
            
        await super().start_consuming()


# Example usage
async def example_usage():
    """Example usage of the message queue."""
    # Create a market data queue
    mq = MarketDataQueue(
        symbol="BTC/USD",
        data_type="tick",
        url="amqp://guest:guest@localhost/"
    )
    
    # Define a message handler
    def handle_market_data(data: Dict) -> None:
        print(f"Received market data: {data}")
    
    # Add the handler
    mq.add_market_data_handler(handle_market_data)
    
    # Connect and start consuming
    async with mq:
        # Publish some test data
        await mq.publish_market_data({
            'price': 50000.0,
            'size': 1.0,
            'side': 'buy'
        })
        
        # Keep running for a while
        await asyncio.sleep(10)

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    asyncio.run(example_usage())
