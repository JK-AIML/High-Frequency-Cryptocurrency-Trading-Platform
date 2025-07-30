"""
Enhanced Redis client with connection pooling and async support.
"""
import redis.asyncio as redis
from typing import Any, Optional, Dict, List, Union, Callable, TypeVar, cast
from functools import wraps
import logging
import json
import pickle
from datetime import timedelta
import asyncio

logger = logging.getLogger(__name__)
F = TypeVar('F', bound=Callable[..., Any])

class RedisManager:
    """Redis connection and cache management with connection pooling."""
    
    _instance = None
    _pool = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(RedisManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, host: str = "localhost", port: int = 6379, 
                 db: int = 0, password: str = None, 
                 max_connections: int = 10, **kwargs):
        if not hasattr(self, '_initialized') or not self._initialized:
            self.host = host
            self.port = port
            self.db = db
            self.password = password
            self.max_connections = max_connections
            self._pool = None
            self._initialized = True
            self._lock = asyncio.Lock()
    
    async def get_pool(self):
        """Get or create a connection pool."""
        if not self._pool:
            self._pool = redis.ConnectionPool(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                max_connections=self.max_connections,
                decode_responses=False
            )
        return self._pool
    
    async def get_client(self):
        """Get a Redis client from the pool."""
        pool = await self.get_pool()
        return redis.Redis(connection_pool=pool)
    
    async def close(self):
        """Close all connections in the pool."""
        if self._pool:
            await self._pool.disconnect()
            self._pool = None
    
    # Cache operations
    async def get(self, key: str) -> Any:
        """Get a value from cache."""
        try:
            client = await self.get_client()
            value = await client.get(key)
            if value is not None:
                return pickle.loads(value)
            return None
        except Exception as e:
            logger.error(f"Redis get error for key {key}: {str(e)}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in cache with optional TTL."""
        try:
            client = await self.get_client()
            serialized = pickle.dumps(value)
            if ttl is not None:
                return await client.setex(key, ttl, serialized)
            return await client.set(key, serialized)
        except Exception as e:
            logger.error(f"Redis set error for key {key}: {str(e)}")
            return False
    
    async def delete(self, *keys: str) -> int:
        """Delete one or more keys from cache."""
        try:
            client = await self.get_client()
            return await client.delete(*keys)
        except Exception as e:
            logger.error(f"Redis delete error for keys {keys}: {str(e)}")
            return 0
    
    async def clear(self, pattern: str = "*") -> int:
        """Clear all keys matching pattern."""
        try:
            client = await self.get_client()
            keys = await client.keys(pattern)
            if keys:
                return await client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Error clearing cache with pattern {pattern}: {str(e)}")
            return 0
    
    # Decorator for caching function results
    def cached(self, ttl: int = 300, key_prefix: str = "cache"):
        """Decorator to cache function results with TTL."""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Create a cache key
                cache_key = f"{key_prefix}:{func.__module__}:{func.__name__}:{args}:{kwargs}"
                
                # Try to get from cache
                cached = await self.get(cache_key)
                if cached is not None:
                    logger.debug(f"Cache hit for {cache_key}")
                    return cached
                
                # Call the function and cache the result
                logger.debug(f"Cache miss for {cache_key}")
                result = await func(*args, **kwargs)
                await self.set(cache_key, result, ttl=ttl)
                return result
            return wrapper
        return decorator

# Global instance
redis_manager = RedisManager()

# For backward compatibility
async def get_redis() -> RedisManager:
    """Get the Redis manager instance."""
    return redis_manager
