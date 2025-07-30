"""
Cache Module

This module provides Redis-based caching capabilities for the data pipeline,
including caching strategies, invalidation, and synchronization.
"""

import logging
import json
import asyncio
from typing import Dict, List, Any, Optional, Union, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
import aioredis
from aioredis.exceptions import RedisError
import backoff

logger = logging.getLogger(__name__)

@dataclass
class CacheConfig:
    """Configuration for Redis cache."""
    host: str = 'localhost'
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    ssl: bool = False
    pool_size: int = 10
    timeout: int = 5
    retry_attempts: int = 3
    retry_delay: int = 1
    default_ttl: int = 3600  # 1 hour

class CacheStrategy:
    """Base class for cache strategies."""
    
    def __init__(self, ttl: Optional[int] = None):
        """
        Initialize cache strategy.
        
        Args:
            ttl: Time-to-live in seconds
        """
        self.ttl = ttl
    
    async def get(self, cache: 'RedisCache', key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            cache: Cache instance
            key: Cache key
            
        Returns:
            Cached value or None
        """
        raise NotImplementedError
    
    async def set(self, cache: 'RedisCache', key: str, value: Any) -> bool:
        """
        Set value in cache.
        
        Args:
            cache: Cache instance
            key: Cache key
            value: Value to cache
            
        Returns:
            True if successful
        """
        raise NotImplementedError
    
    async def delete(self, cache: 'RedisCache', key: str) -> bool:
        """
        Delete value from cache.
        
        Args:
            cache: Cache instance
            key: Cache key
            
        Returns:
            True if successful
        """
        raise NotImplementedError

class LRUStrategy(CacheStrategy):
    """Least Recently Used cache strategy."""
    
    async def get(self, cache: 'RedisCache', key: str) -> Optional[Any]:
        """Get value using LRU strategy."""
        try:
            value = await cache.redis.get(key)
            if value:
                # Update last accessed time
                await cache.redis.zadd('lru:keys', {key: datetime.utcnow().timestamp()})
            return value
        except RedisError as e:
            logger.error(f"Error getting value from cache: {e}")
            return None
    
    async def set(self, cache: 'RedisCache', key: str, value: Any) -> bool:
        """Set value using LRU strategy."""
        try:
            # Set value
            await cache.redis.set(key, value, ex=self.ttl)
            
            # Update last accessed time
            await cache.redis.zadd('lru:keys', {key: datetime.utcnow().timestamp()})
            
            # Check cache size
            size = await cache.redis.zcard('lru:keys')
            if size > cache.max_size:
                # Remove least recently used
                await cache.redis.zremrangebyrank('lru:keys', 0, 0)
            
            return True
        except RedisError as e:
            logger.error(f"Error setting value in cache: {e}")
            return False
    
    async def delete(self, cache: 'RedisCache', key: str) -> bool:
        """Delete value using LRU strategy."""
        try:
            await cache.redis.delete(key)
            await cache.redis.zrem('lru:keys', key)
            return True
        except RedisError as e:
            logger.error(f"Error deleting value from cache: {e}")
            return False

class RedisCache:
    """Redis-based cache implementation."""
    
    def __init__(self, config: CacheConfig, strategy: Optional[CacheStrategy] = None):
        """
        Initialize Redis cache.
        
        Args:
            config: Cache configuration
            strategy: Optional cache strategy
        """
        self.config = config
        self.strategy = strategy or LRUStrategy(ttl=config.default_ttl)
        self.redis = None
        self.max_size = 10000  # Maximum number of cached items
        self._connected = False
    
    @backoff.on_exception(backoff.expo, RedisError, max_tries=3)
    async def connect(self) -> None:
        """Establish Redis connection."""
        if self._connected:
            return
            
        try:
            self.redis = await aioredis.create_redis_pool(
                f'redis://{self.config.host}:{self.config.port}',
                db=self.config.db,
                password=self.config.password,
                ssl=self.config.ssl,
                minsize=1,
                maxsize=self.config.pool_size,
                timeout=self.config.timeout
            )
            self._connected = True
            logger.info(f"Connected to Redis at {self.config.host}:{self.config.port}")
            
        except RedisError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._connected = False
            raise
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        if not self._connected:
            await self.connect()
            
        return await self.strategy.get(self, key)
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional time-to-live in seconds
            
        Returns:
            True if successful
        """
        if not self._connected:
            await self.connect()
            
        if ttl is not None:
            self.strategy.ttl = ttl
            
        return await self.strategy.set(self, key, value)
    
    async def delete(self, key: str) -> bool:
        """
        Delete value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if successful
        """
        if not self._connected:
            await self.connect()
            
        return await self.strategy.delete(self, key)
    
    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key exists
        """
        if not self._connected:
            await self.connect()
            
        try:
            return await self.redis.exists(key)
        except RedisError as e:
            logger.error(f"Error checking key existence: {e}")
            return False
    
    async def clear(self) -> bool:
        """
        Clear all cached values.
        
        Returns:
            True if successful
        """
        if not self._connected:
            await self.connect()
            
        try:
            await self.redis.flushdb()
            return True
        except RedisError as e:
            logger.error(f"Error clearing cache: {e}")
            return False
    
    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """
        Get multiple values from cache.
        
        Args:
            keys: List of cache keys
            
        Returns:
            Dictionary mapping keys to values
        """
        if not self._connected:
            await self.connect()
            
        try:
            values = await self.redis.mget(keys)
            return {k: v for k, v in zip(keys, values) if v is not None}
        except RedisError as e:
            logger.error(f"Error getting multiple values: {e}")
            return {}
    
    async def set_many(self, mapping: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """
        Set multiple values in cache.
        
        Args:
            mapping: Dictionary mapping keys to values
            ttl: Optional time-to-live in seconds
            
        Returns:
            True if successful
        """
        if not self._connected:
            await self.connect()
            
        try:
            pipeline = self.redis.pipeline()
            for key, value in mapping.items():
                pipeline.set(key, value, ex=ttl or self.strategy.ttl)
            await pipeline.execute()
            return True
        except RedisError as e:
            logger.error(f"Error setting multiple values: {e}")
            return False
    
    async def delete_many(self, keys: List[str]) -> bool:
        """
        Delete multiple values from cache.
        
        Args:
            keys: List of cache keys
            
        Returns:
            True if successful
        """
        if not self._connected:
            await self.connect()
            
        try:
            await self.redis.delete(*keys)
            return True
        except RedisError as e:
            logger.error(f"Error deleting multiple values: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary of cache statistics
        """
        if not self._connected:
            await self.connect()
            
        try:
            info = await self.redis.info()
            return {
                'connected_clients': info['connected_clients'],
                'used_memory': info['used_memory'],
                'total_keys': info['db0']['keys'],
                'hits': info['keyspace_hits'],
                'misses': info['keyspace_misses']
            }
        except RedisError as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}
    
    async def close(self) -> None:
        """Close Redis connection."""
        if self.redis:
            self.redis.close()
            await self.redis.wait_closed()
        self._connected = False 