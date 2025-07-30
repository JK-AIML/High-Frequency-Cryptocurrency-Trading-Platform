"""
Test suite for real-time data streaming components.
"""

import pytest
import asyncio
import json
from datetime import datetime
from src.tick_analysis.data_pipeline.streaming import (
    WebSocketConfig,
    KafkaConfig,
    RabbitMQConfig,
    WebSocketStream,
    KafkaStream,
    RabbitMQStream
)

@pytest.fixture
def test_data():
    """Generate test data."""
    return {
        'timestamp': datetime.utcnow().isoformat(),
        'symbol': 'AAPL',
        'price': 150.0,
        'volume': 1000,
        'bid': 149.9,
        'ask': 150.1
    }

@pytest.fixture
def websocket_config():
    """Create WebSocket configuration."""
    return WebSocketConfig(
        url='ws://localhost:8080',
        protocols=['v1'],
        headers={'Authorization': 'Bearer test-token'}
    )

@pytest.fixture
def kafka_config():
    """Create Kafka configuration."""
    return KafkaConfig(
        bootstrap_servers=['localhost:9092'],
        topic='tick_data',
        group_id='test-group',
        client_id='test-client'
    )

@pytest.fixture
def rabbitmq_config():
    """Create RabbitMQ configuration."""
    return RabbitMQConfig(
        host='localhost',
        port=5672,
        username='guest',
        password='guest',
        exchange='tick_data',
        queue='tick_data_queue'
    )

@pytest.mark.asyncio
async def test_websocket_stream(websocket_config, test_data):
    """Test WebSocket streaming."""
    # Initialize stream
    stream = WebSocketStream(websocket_config)
    
    # Mock WebSocket server
    async def mock_server(websocket, path):
        await websocket.send(json.dumps(test_data))
    
    # Start mock server
    server = await websockets.serve(mock_server, 'localhost', 8080)
    
    try:
        # Connect to WebSocket
        await stream.connect()
        
        # Subscribe to channel
        await stream.subscribe(['tick_data'])
        
        # Add message handler
        received_data = []
        async def handler(data):
            received_data.append(data)
        stream.add_handler(handler)
        
        # Start receiving messages
        await asyncio.sleep(1)  # Wait for message
        
        # Verify received data
        assert len(received_data) > 0
        assert received_data[0]['symbol'] == test_data['symbol']
        
    finally:
        # Cleanup
        await stream.close()
        server.close()
        await server.wait_closed()

@pytest.mark.asyncio
async def test_kafka_stream(kafka_config, test_data):
    """Test Kafka streaming."""
    # Initialize stream
    stream = KafkaStream(kafka_config)
    
    try:
        # Connect to Kafka
        await stream.connect()
        
        # Add message handler
        received_data = []
        async def handler(data):
            received_data.append(data)
        stream.add_handler(handler)
        
        # Produce message
        await stream.produce(test_data)
        
        # Start consuming messages
        await asyncio.sleep(1)  # Wait for message
        
        # Verify received data
        assert len(received_data) > 0
        assert received_data[0]['symbol'] == test_data['symbol']
        
    finally:
        # Cleanup
        await stream.close()

@pytest.mark.asyncio
async def test_rabbitmq_stream(rabbitmq_config, test_data):
    """Test RabbitMQ streaming."""
    # Initialize stream
    stream = RabbitMQStream(rabbitmq_config)
    
    try:
        # Connect to RabbitMQ
        await stream.connect()
        
        # Add message handler
        received_data = []
        async def handler(data):
            received_data.append(data)
        stream.add_handler(handler)
        
        # Produce message
        await stream.produce(test_data)
        
        # Start consuming messages
        await asyncio.sleep(1)  # Wait for message
        
        # Verify received data
        assert len(received_data) > 0
        assert received_data[0]['symbol'] == test_data['symbol']
        
    finally:
        # Cleanup
        await stream.close()

@pytest.mark.asyncio
async def test_stream_error_handling(websocket_config, kafka_config, rabbitmq_config):
    """Test error handling in streams."""
    # Test WebSocket error handling
    stream = WebSocketStream(websocket_config)
    with pytest.raises(Exception):
        await stream.connect()
    await stream.close()
    
    # Test Kafka error handling
    stream = KafkaStream(kafka_config)
    with pytest.raises(Exception):
        await stream.connect()
    await stream.close()
    
    # Test RabbitMQ error handling
    stream = RabbitMQStream(rabbitmq_config)
    with pytest.raises(Exception):
        await stream.connect()
    await stream.close()

@pytest.mark.asyncio
async def test_stream_reconnection(websocket_config, test_data):
    """Test stream reconnection."""
    # Initialize stream
    stream = WebSocketStream(websocket_config)
    
    # Mock WebSocket server with connection drop
    connected = True
    async def mock_server(websocket, path):
        nonlocal connected
        if connected:
            await websocket.send(json.dumps(test_data))
            connected = False
            await websocket.close()
        else:
            connected = True
            await websocket.send(json.dumps(test_data))
    
    # Start mock server
    server = await websockets.serve(mock_server, 'localhost', 8080)
    
    try:
        # Connect to WebSocket
        await stream.connect()
        
        # Add message handler
        received_data = []
        async def handler(data):
            received_data.append(data)
        stream.add_handler(handler)
        
        # Start receiving messages
        await asyncio.sleep(2)  # Wait for reconnection
        
        # Verify received data
        assert len(received_data) > 0
        assert received_data[0]['symbol'] == test_data['symbol']
        
    finally:
        # Cleanup
        await stream.close()
        server.close()
        await server.wait_closed()

@pytest.mark.asyncio
async def test_stream_message_handling(websocket_config, test_data):
    """Test message handling in streams."""
    # Initialize stream
    stream = WebSocketStream(websocket_config)
    
    # Mock WebSocket server
    async def mock_server(websocket, path):
        await websocket.send(json.dumps(test_data))
        await websocket.send(json.dumps({'invalid': 'data'}))
    
    # Start mock server
    server = await websockets.serve(mock_server, 'localhost', 8080)
    
    try:
        # Connect to WebSocket
        await stream.connect()
        
        # Add message handlers
        received_data = []
        errors = []
        
        async def handler(data):
            if 'symbol' not in data:
                errors.append('Invalid data format')
                return
            received_data.append(data)
            
        stream.add_handler(handler)
        
        # Start receiving messages
        await asyncio.sleep(1)  # Wait for messages
        
        # Verify received data and errors
        assert len(received_data) > 0
        assert len(errors) > 0
        assert received_data[0]['symbol'] == test_data['symbol']
        
    finally:
        # Cleanup
        await stream.close()
        server.close()
        await server.wait_closed() 