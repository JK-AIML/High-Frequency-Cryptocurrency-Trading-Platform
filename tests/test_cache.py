"""
Test suite for Redis cache implementation.
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from src.tick_analysis.data_pipeline.cache import (
    CacheConfig,
    CacheStrategy,
    LRUStrategy,
    RedisCache
)

@pytest.fixture
def cache_config():
    """Create cache configuration."""
    return CacheConfig(
        host='localhost',
        port=6379,
        db=0,
        password=None,
        ssl=False,
        pool_size=5,
        timeout=5,
        default_ttl=3600
    )

@pytest.fixture
def test_data():
    """Generate test data."""
    return {
        'key1': 'value1',
        'key2': 'value2',
        'key3': 'value3'
    }

@pytest.mark.asyncio
async def test_cache_operations(cache_config, test_data):
    """Test basic cache operations."""
    # Initialize cache
    cache = RedisCache(cache_config)
    
    try:
        # Connect to Redis
        await cache.connect()
        
        # Test set and get
        for key, value in test_data.items():
            await cache.set(key, value)
            cached_value = await cache.get(key)
            assert cached_value == value
        
        # Test exists
        assert await cache.exists('key1')
        assert not await cache.exists('nonexistent')
        
        # Test delete
        await cache.delete('key1')
        assert not await cache.exists('key1')
        
        # Test clear
        await cache.clear()
        assert not await cache.exists('key2')
        assert not await cache.exists('key3')
        
    finally:
        # Cleanup
        await cache.close()

@pytest.mark.asyncio
async def test_bulk_operations(cache_config, test_data):
    """Test bulk cache operations."""
    # Initialize cache
    cache = RedisCache(cache_config)
    
    try:
        # Connect to Redis
        await cache.connect()
        
        # Test set_many
        await cache.set_many(test_data)
        
        # Test get_many
        values = await cache.get_many(list(test_data.keys()))
        assert values == test_data
        
        # Test delete_many
        await cache.delete_many(list(test_data.keys()))
        values = await cache.get_many(list(test_data.keys()))
        assert not values
        
    finally:
        # Cleanup
        await cache.close()

@pytest.mark.asyncio
async def test_lru_strategy(cache_config):
    """Test LRU cache strategy."""
    # Initialize cache with LRU strategy
    strategy = LRUStrategy(ttl=3600)
    cache = RedisCache(cache_config, strategy)
    
    try:
        # Connect to Redis
        await cache.connect()
        
        # Fill cache
        for i in range(100):
            await cache.set(f'key{i}', f'value{i}')
        
        # Access some keys
        await cache.get('key0')
        await cache.get('key1')
        
        # Check LRU order
        lru_keys = await cache.redis.zrange('lru:keys', 0, -1)
        assert lru_keys[-1] == b'key0'  # Most recently used
        assert lru_keys[-2] == b'key1'  # Second most recently used
        
    finally:
        # Cleanup
        await cache.close()

@pytest.mark.asyncio
async def test_ttl(cache_config):
    """Test time-to-live functionality."""
    # Initialize cache with short TTL
    strategy = LRUStrategy(ttl=1)
    cache = RedisCache(cache_config, strategy)
    
    try:
        # Connect to Redis
        await cache.connect()
        
        # Set value with TTL
        await cache.set('key', 'value')
        assert await cache.exists('key')
        
        # Wait for TTL
        await asyncio.sleep(2)
        assert not await cache.exists('key')
        
    finally:
        # Cleanup
        await cache.close()

@pytest.mark.asyncio
async def test_error_handling(cache_config):
    """Test error handling."""
    # Initialize cache with invalid config
    invalid_config = CacheConfig(host='invalid_host')
    cache = RedisCache(invalid_config)
    
    # Test connection error
    with pytest.raises(Exception):
        await cache.connect()
    
    # Test operations with no connection
    assert not await cache.set('key', 'value')
    assert not await cache.get('key')
    assert not await cache.delete('key')
    assert not await cache.exists('key')
    assert not await cache.clear()
    assert not await cache.set_many({'key': 'value'})
    assert not await cache.get_many(['key'])
    assert not await cache.delete_many(['key'])

@pytest.mark.asyncio
async def test_cache_stats(cache_config, test_data):
    """Test cache statistics."""
    # Initialize cache
    cache = RedisCache(cache_config)
    
    try:
        # Connect to Redis
        await cache.connect()
        
        # Add some data
        await cache.set_many(test_data)
        
        # Get stats
        stats = await cache.get_stats()
        assert 'connected_clients' in stats
        assert 'used_memory' in stats
        assert 'total_keys' in stats
        assert 'hits' in stats
        assert 'misses' in stats
        
    finally:
        # Cleanup
        await cache.close()

@pytest.mark.asyncio
async def test_cache_pooling(cache_config):
    """Test connection pooling."""
    # Initialize cache with pool size 2
    config = CacheConfig(pool_size=2)
    cache = RedisCache(config)
    
    try:
        # Connect to Redis
        await cache.connect()
        
        # Create multiple concurrent operations
        async def operation(i):
            await cache.set(f'key{i}', f'value{i}')
            await cache.get(f'key{i}')
        
        # Run operations concurrently
        await asyncio.gather(*[operation(i) for i in range(10)])
        
        # Check pool size
        assert cache.redis._pool.size <= config.pool_size
        
    finally:
        # Cleanup
        await cache.close()

@pytest.mark.asyncio
async def test_cache_serialization(cache_config):
    """Test data serialization."""
    # Initialize cache
    cache = RedisCache(cache_config)
    
    try:
        # Connect to Redis
        await cache.connect()
        
        # Test different data types
        test_data = {
            'string': 'value',
            'int': 42,
            'float': 3.14,
            'bool': True,
            'list': [1, 2, 3],
            'dict': {'a': 1, 'b': 2},
            'datetime': datetime.utcnow()
        }
        
        # Store data
        for key, value in test_data.items():
            await cache.set(key, json.dumps(value))
        
        # Retrieve data
        for key, value in test_data.items():
            cached_value = json.loads(await cache.get(key))
            if isinstance(value, datetime):
                assert datetime.fromisoformat(cached_value) == value
            else:
                assert cached_value == value
        
    finally:
        # Cleanup
        await cache.close() 