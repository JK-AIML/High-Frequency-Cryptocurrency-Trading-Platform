"""
Redis caching layer for the Tick Data Analysis system.
"""
import redis
from functools import wraps
import pickle
import json
from typing import Any, Callable, Optional, TypeVar, cast
from datetime import timedelta
import logging

# Type variable for generic function typing
F = TypeVar('F', bound=Callable[..., Any])

logger = logging.getLogger(__name__)

class RedisCache:
    """Redis-based caching system with automatic serialization/deserialization."""
    
    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0, 
                 password: Optional[str] = None, **kwargs):
        """Initialize Redis connection.
        
        Args:
            host: Redis server host
            port: Redis server port
            db: Redis database number
            password: Redis authentication password
            **kwargs: Additional Redis connection parameters
        """
        self.redis = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=False,  # We'll handle encoding/decoding ourselves
            **kwargs
        )
        
        # Test connection
        try:
            self.redis.ping()
            logger.info("Successfully connected to Redis")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def get(self, key: str) -> Any:
        """Get a value from cache."""
        try:
            value = self.redis.get(key)
            if value is None:
                return None
            return pickle.loads(value)
        except (pickle.PickleError, redis.RedisError) as e:
            logger.error(f"Error getting key {key}: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in cache with optional TTL."""
        try:
            serialized = pickle.dumps(value)
            if ttl is not None:
                return self.redis.setex(key, ttl, serialized)
            return self.redis.set(key, serialized)
        except (pickle.PickleError, redis.RedisError) as e:
            logger.error(f"Error setting key {key}: {e}")
            return False
    
    def delete(self, *keys: str) -> int:
        """Delete one or more keys from cache."""
        try:
            return self.redis.delete(*keys)
        except redis.RedisError as e:
            logger.error(f"Error deleting keys {keys}: {e}")
            return 0
    
    def clear(self, pattern: str = "*") -> int:
        """Clear all keys matching pattern."""
        try:
            keys = self.redis.keys(pattern)
            if keys:
                return self.redis.delete(*keys)
            return 0
        except redis.RedisError as e:
            logger.error(f"Error clearing cache with pattern {pattern}: {e}")
            return 0
    
    def cache(self, ttl: int = 300, key_prefix: str = "cache") -> Callable[[F], F]:
        """Decorator to cache function results.
        
        Args:
            ttl: Time to live in seconds
            key_prefix: Prefix for cache keys
            
        Returns:
            Decorated function with caching
        """
        def decorator(func: F) -> F:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Create a cache key based on function name and arguments
                cache_key = f"{key_prefix}:{func.__module__}:{func.__name__}:{args}:{kwargs}"
                
                # Try to get from cache
                cached = self.get(cache_key)
                if cached is not None:
                    logger.debug(f"Cache hit for {cache_key}")
                    return cached
                
                # Call the function and cache the result
                logger.debug(f"Cache miss for {cache_key}")
                result = func(*args, **kwargs)
                self.set(cache_key, result, ttl=ttl)
                return result
            return cast(F, wrapper)
        return decorator

# Create a default instance
cache = RedisCache(
    host='localhost',
    port=6379,
    db=0,
    password=None
)

def get_redis() -> RedisCache:
    """Get the default Redis cache instance."""
    return cache
