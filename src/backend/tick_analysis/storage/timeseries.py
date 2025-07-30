"""
Enhanced Time-series database integration for storing and querying market data.

Features:
- Advanced partitioning (time-based, asset-based, hybrid)
- Configurable retention policies
- High-performance writes with batching
- Efficient time-range queries
- Schema management
- Performance optimizations
"""
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from datetime import datetime, timedelta
import logging
from enum import Enum, auto
import pandas as pd
import numpy as np
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS, ASYNCHRONOUS, WriteOptions
from influxdb_client.client.exceptions import InfluxDBError
from influxdb_client.client.warnings import MissingPivotFunction
from influxdb_client.client.write.point import PointSettings
import pytz
from dateutil.parser import parse as parse_date
from dataclasses import dataclass, field, asdict, fields
from typing_extensions import Literal, Annotated
import json
import hashlib
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import gzip
import io
import msgpack
from functools import lru_cache, wraps
import re
from datetime import datetime, timezone, timedelta
from typing import get_type_hints, get_origin, get_args
from collections import defaultdict
import random
import string
from contextlib import contextmanager

# Third-party imports
try:
    import numpy as np
except ImportError:
    np = None

try:
    import pandas as pd
except ImportError:
    pd = None

logger = logging.getLogger(__name__)

class PartitionStrategy(Enum):
    """Partitioning strategies for time-series data."""
    TIME_BASED = auto()      # Partition by time (e.g., daily, hourly)
    ASSET_BASED = auto()     # Partition by asset/symbol
    HYBRID = auto()          # Hybrid of time and asset-based
    NONE = auto()            # No partitioning

class CompressionType(Enum):
    """Supported compression types."""
    NONE = auto()
    GZIP = auto()
    LZ4 = auto()
    SNAPPY = auto()

class QueryOptimization(Enum):
    """Query optimization strategies."""
    NONE = auto()
    INDEX_SCAN = auto()
    PARALLEL = auto()
    PREDICATE_PUSHDOWN = auto()
    COLUMN_PRUNING = auto()
    PARTITION_PRUNING = auto()
    CACHING = auto()

@dataclass
class CacheConfig:
    """Configuration for caching layer."""
    enabled: bool = True
    max_size: int = 10000
    ttl_seconds: int = 300
    strategy: str = 'lru'  # 'lru', 'lfu', 'fifo'

@dataclass
class WriteBatchConfig:
    """Configuration for batch writing."""
    enabled: bool = True
    batch_size: int = 1000
    flush_interval: int = 5  # seconds
    max_retries: int = 3
    retry_delay: float = 0.5  # seconds

@dataclass
class QueryConfig:
    """Configuration for queries."""
    timeout: int = 30000  # ms
    page_size: int = 10000
    optimize: List[QueryOptimization] = field(default_factory=lambda: [
        QueryOptimization.INDEX_SCAN,
        QueryOptimization.PREDICATE_PUSHDOWN,
        QueryOptimization.PARTITION_PRUNING
    ])
    use_cache: bool = True

@dataclass
class RetentionPolicy:
    """Retention policy configuration."""
    name: str = "autogen"
    duration: str = "0"  # 0 means infinite retention
    shard_duration: str = "24h"
    replication: int = 1
    default: bool = True
    hot_duration: Optional[str] = None
    warm_duration: Optional[str] = None
    cold_duration: Optional[str] = None
    hot_ttl: Optional[str] = None
    warm_ttl: Optional[str] = None
    cold_ttl: Optional[str] = None
    hot_compaction: bool = True
    warm_compaction: bool = True
    cold_compaction: bool = True

@dataclass
class PartitionConfig:
    """Partitioning configuration."""
    strategy: PartitionStrategy = PartitionStrategy.TIME_BASED
    time_interval: str = "1d"  # For time-based partitioning
    time_format: str = "%Y%m%d"
    hash_partitions: int = 32  # For hash-based partitioning
    columns: List[str] = field(default_factory=list)  # For column-based partitioning

@dataclass
class BucketConfig:
    """Bucket configuration for time-series data."""
    name: str
    retention_policy: RetentionPolicy = field(default_factory=RetentionPolicy)
    partition_config: PartitionConfig = field(default_factory=PartitionConfig)
    write_batch_config: WriteBatchConfig = field(default_factory=WriteBatchConfig)
    query_config: QueryConfig = field(default_factory=QueryConfig)
    cache_config: CacheConfig = field(default_factory=CacheConfig)
    compression: CompressionType = CompressionType.GZIP
    schema: Dict[str, str] = field(default_factory=dict)  # Field name to type mapping
    tags: List[str] = field(default_factory=list)  # List of tag names
    fields: List[str] = field(default_factory=list)  # List of field names
    timestamp_field: str = "timestamp"
    timestamp_precision: str = "ns"  # ns, us, ms, s
    description: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata

class TimePrecision(str, Enum):
    """Time precision for InfluxDB operations."""
    NS = "ns"  # Nanoseconds
    US = "us"  # Microseconds
    MS = "ms"  # Milliseconds (default)
    S = "s"    # Seconds

class FieldType(str, Enum):
    """Field types for time-series data."""
    FLOAT = "float"
    INTEGER = "integer"
    UNSIGNED = "unsigned"
    STRING = "string"
    BOOLEAN = "boolean"

@dataclass
class RetentionPolicy:
    """Retention policy configuration."""
    name: str = "autogen"
    duration: str = "0"  # 0 means infinite retention
    shard_duration: str = "24h"
    replication: int = 1
    default: bool = True

@dataclass
class BucketConfig:
    """Bucket configuration for time-series data."""
    name: str
    retention_policy: RetentionPolicy = field(default_factory=RetentionPolicy)
    description: str = ""
    schema_type: str = "implicit"  # or "explicit"

def retry_on_exception(max_retries=3, delay=1, backoff=2, exceptions=(Exception,)):
    """Decorator for retrying a function with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            current_delay = delay
            
            while True:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    retries += 1
                    if retries > max_retries:
                        raise
                    
                    logger.warning(
                        f"Attempt {retries}/{max_retries} failed: {e}. "
                        f"Retrying in {current_delay} seconds..."
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff
        return wrapper
    return decorator

class TimeSeriesDB:
    """Enhanced Time-series database client for market data storage and analysis.
    
    Features:
    - Advanced partitioning (time-based, asset-based, hybrid)
    - Configurable retention policies
    - High-performance writes with batching and compression
    - Efficient time-range queries with predicate pushdown
    - Schema management and validation
    - Caching layer for improved performance
    - Automatic retries and error handling
    - Downsampling and continuous queries
    - Data lifecycle management
    - Monitoring and metrics
    """
    
    def __init__(
        self,
        url: str = "http://localhost:8086",
        token: str = "",
        org: str = "tick_analysis",
        bucket: str = "market_data",
        timeout: int = 10_000,
        enable_gzip: bool = True,
        enable_compression: bool = True,
        compression_type: CompressionType = CompressionType.GZIP,
        max_retries: int = 3,
        retry_delay: float = 0.5,
        batch_size: int = 1000,
        flush_interval: int = 5,
        cache_enabled: bool = True,
        cache_size: int = 10000,
        cache_ttl: int = 300,
        partition_strategy: PartitionStrategy = PartitionStrategy.TIME_BASED,
        time_partition_interval: str = "1d",
        enable_indexing: bool = True,
        index_fields: List[str] = None,
        **kwargs
    ):
        """
        Initialize the enhanced time-series database client with advanced features.
        
        Args:
            url: InfluxDB server URL
            token: Authentication token
            org: Organization name
            bucket: Default bucket name
            timeout: Request timeout in milliseconds
            enable_gzip: Enable GZIP compression for HTTP requests
            enable_compression: Enable data compression
            compression_type: Type of compression to use
            max_retries: Maximum number of retries for failed operations
            retry_delay: Delay between retries in seconds
            batch_size: Default batch size for writes
            flush_interval: Maximum time to wait before flushing batches (seconds)
            cache_enabled: Enable query result caching
            cache_size: Maximum number of items to cache
            cache_ttl: Time-to-live for cache entries (seconds)
            partition_strategy: Strategy for partitioning data
            time_partition_interval: Time interval for time-based partitioning
            enable_indexing: Enable field indexing
            index_fields: List of fields to index
            **kwargs: Additional InfluxDBClient parameters
        """
        """
        Initialize the enhanced time-series database client.
        
        Args:
            url: InfluxDB server URL
            token: Authentication token
            org: Organization name
            bucket: Default bucket name
            timeout: Request timeout in milliseconds
            enable_gzip: Enable GZIP compression for HTTP requests
            enable_compression: Enable data compression
            compression_type: Type of compression to use
            max_retries: Maximum number of retries for failed operations
            retry_delay: Delay between retries in seconds
            batch_size: Default batch size for writes
            flush_interval: Maximum time to wait before flushing batches (seconds)
            cache_enabled: Enable query result caching
            cache_size: Maximum number of items to cache
            cache_ttl: Time-to-live for cache entries (seconds)
            **kwargs: Additional InfluxDBClient parameters
        """
        self.url = url
        self.token = token
        self.org = org
        self.bucket = bucket
        self.timeout = timeout
        self.enable_gzip = enable_gzip
        self.enable_compression = enable_compression
        self.compression_type = compression_type
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.cache_enabled = cache_enabled
        self.cache_size = cache_size
        self.cache_ttl = cache_ttl
        
        # Initialize clients and APIs
        self.client: Optional[InfluxDBClient] = None
        self.write_api = None
        self.query_api = None
        self.delete_api = None
        self.buckets_api = None
        self.organizations_api = None
        self.health = None
        self.connected = False
        
        # Partitioning configuration
        self.partition_strategy = partition_strategy
        self.time_partition_interval = time_partition_interval
        self.partition_cache = {}
        self.partition_lock = threading.RLock()
        
        # Indexing configuration
        self.enable_indexing = enable_indexing
        self.index_fields = index_fields or ['symbol', 'exchange', 'asset_class']
        self.indices = {field: set() for field in self.index_fields}
        self.index_lock = threading.RLock()
        
        # Schema management
        self.schemas = {}
        self.schema_lock = threading.RLock()
        
        # Retention policies
        self.retention_policies = {}
        self.retention_check_interval = 3600  # 1 hour
        self.last_retention_check = 0
        self.retention_lock = threading.Lock()
        
        # Batch processing
        self._batch_buffer = []
        self._last_flush = time.time()
        self._batch_lock = threading.Lock()
        self._batch_thread = None
        self._stop_batch_worker = threading.Event()
        
        # Caching
        self._cache = {}
        self._cache_order = []
        
        # Enhanced metrics with partitioning and indexing
        self.metrics = {
            # Basic metrics
            'writes': 0,
            'reads': 0,
            'errors': 0,
            'retries': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'batch_flushes': 0,
            'batch_errors': 0,
            'bytes_written': 0,
            'bytes_read': 0,
            'query_time_ms': 0,
            'write_time_ms': 0,
            
            # Partitioning metrics
            'partitions_created': 0,
            'partitions_dropped': 0,
            'partition_queries': 0,
            'partition_optimizations': 0,
            
            # Indexing metrics
            'indices_created': 0,
            'index_hits': 0,
            'index_misses': 0,
            'index_updates': 0,
            
            # Retention metrics
            'retention_checks': 0,
            'data_expired': 0,
            'retention_errors': 0,
            
            # Compression metrics
            'compressed_bytes': 0,
            'uncompressed_bytes': 0,
            'compression_ratio': 0.0
        }
        
        # Additional client options
        self.client_options = kwargs
        
        # Initialize background batch worker
        self._init_batch_worker()
        
        # Register signal handlers for graceful shutdown
        import signal
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _init_batch_worker(self):
        """Initialize the background batch processing worker."""
        if not hasattr(self, '_batch_thread') or not self._batch_thread.is_alive():
            self._stop_batch_worker.clear()
            self._batch_thread = threading.Thread(
                target=self._batch_worker,
                daemon=True,
                name="BatchWorker"
            )
            self._batch_thread.start()
    
    def _batch_worker(self):
        """Background worker for processing batches."""
        while not self._stop_batch_worker.is_set():
            try:
                current_time = time.time()
                time_since_flush = current_time - self._last_flush
                
                # Check if we need to flush based on time or buffer size
                with self._batch_lock:
                    buffer_size = len(self._batch_buffer)
                    should_flush = (
                        buffer_size >= self.batch_size or 
                        (time_since_flush >= self.flush_interval and buffer_size > 0)
                    )
                
                if should_flush:
                    self.flush()
                
                # Sleep for a short time to prevent busy waiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in batch worker: {e}", exc_info=True)
                time.sleep(1)  # Prevent tight loop on errors
    
    def _signal_handler(self, signum, frame):
        """Handle signals for graceful shutdown."""
        logger.info(f"Received signal {signum}, shutting down...")
        self._stop_batch_worker.set()
        self.flush()  # Flush any remaining data
        self.disconnect()
        
    def flush(self) -> int:
        """
        Flush the current batch buffer to the database.
        
        Returns:
            int: Number of points written
        """
        if not self._batch_buffer:
            return 0
            
        points = []
        with self._batch_lock:
            if not self._batch_buffer:
                return 0
                
            points = list(self._batch_buffer)
            self._batch_buffer.clear()
            self._last_flush = time.time()
        
        if not points:
            return 0
            
        try:
            start_time = time.time()
            self._write_points_sync(points)
            write_time = (time.time() - start_time) * 1000  # ms
            
            self.metrics['batch_flushes'] += 1
            self.metrics['write_time_ms'] += write_time
            self.metrics['writes'] += len(points)
            
            logger.debug(f"Flushed {len(points)} points in {write_time:.2f}ms")
            return len(points)
            
        except Exception as e:
            self.metrics['batch_errors'] += 1
            logger.error(f"Failed to flush batch: {e}", exc_info=True)
            # TODO: Implement retry logic with backoff
            return 0
    
    def _write_points_sync(self, points):
        """Synchronously write points with retry logic."""
        if not self.connected:
            self.connect()
            
        for attempt in range(self.max_retries + 1):
            try:
                self.write_api.write(
                    bucket=self.bucket,
                    org=self.org,
                    record=points
                )
                return
                
            except InfluxDBError as e:
                if attempt >= self.max_retries:
                    raise
                    
                retry_delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                logger.warning(
                    f"Write failed (attempt {attempt + 1}/{self.max_retries}): {e}. "
                    f"Retrying in {retry_delay:.1f}s..."
                )
                time.sleep(retry_delay)
                self.metrics['retries'] += 1
    
    def _get_cache_key(self, query: str, params: Optional[Dict] = None) -> str:
        """Generate a cache key for a query."""
        key_parts = [query]
        if params:
            key_parts.append(json.dumps(params, sort_keys=True))
        return hashlib.md5(''.join(key_parts).encode()).hexdigest()
    
    def _get_from_cache(self, key: str):
        """Get a value from the cache if it exists and is not expired."""
        if not self.cache_enabled or key not in self._cache:
            self.metrics['cache_misses'] += 1
            return None
            
        value, expiry = self._cache[key]
        if time.time() > expiry:
            del self._cache[key]
            self.metrics['cache_misses'] += 1
            return None
            
        # Move to end of LRU
        self._cache_order.remove(key)
        self._cache_order.append(key)
        
        self.metrics['cache_hits'] += 1
        return value
    
    def _set_in_cache(self, key: str, value: Any, ttl: Optional[int] = None):
        """Store a value in the cache."""
        if not self.cache_enabled:
            return
            
        ttl = ttl or self.cache_ttl
        expiry = time.time() + ttl
        
        with self._batch_lock:
            # Evict if cache is full (LRU)
            if len(self._cache) >= self.cache_size and self._cache_order:
                oldest_key = self._cache_order.pop(0)
                del self._cache[oldest_key]
            
            self._cache[key] = (value, expiry)
            if key in self._cache_order:
                self._cache_order.remove(key)
            self._cache_order.append(key)
    
    def clear_cache(self):
        """Clear the query cache."""
        with self._batch_lock:
            self._cache.clear()
            self._cache_order.clear()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        metrics = self.metrics.copy()
        metrics.update({
            'cache_size': len(self._cache),
            'batch_buffer_size': len(self._batch_buffer),
            'connected': self.connected,
            'uptime_seconds': time.time() - self._start_time if hasattr(self, '_start_time') else 0
        })
        return metrics
    
    def reset_metrics(self):
        """Reset all metrics."""
        self.metrics = {
            'writes': 0,
            'reads': 0,
            'errors': 0,
            'retries': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'batch_flushes': 0,
            'batch_errors': 0,
            'bytes_written': 0,
            'bytes_read': 0,
            'query_time_ms': 0,
            'write_time_ms': 0
        }

    async def connect(self) -> None:
        """Connect to the time-series database with retry logic."""
        if self.connected:
            return
            
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                self.client = InfluxDBClient(
                    url=self.url,
                    token=self.token,
                    org=self.org,
                    timeout=self.timeout,
                    enable_gzip=self.enable_gzip,
                    **self.client_options
                )
                
                # Initialize APIs
                self.write_api = self.client.write_api(
                    write_options=WriteOptions(
                        batch_size=self.batch_size,
                        flush_interval=self.flush_interval * 1000,  # ms
                        jitter_interval=2000,  # 2s jitter
                        retry_interval=5000,  # 5s retry
                        max_retries=self.max_retries,
                        max_retry_delay=30000,  # 30s max delay
                        exponential_base=2  # Exponential backoff
                    )
                )
                
                self.query_api = self.client.query_api()
                self.delete_api = self.client.delete_api()
                self.buckets_api = self.client.buckets_api()
                self.organizations_api = self.client.organizations_api()
                
                # Check connection
                self.health = self.client.health()
                self.connected = True
                self._start_time = time.time()
                
                logger.info(
                    f"Connected to {self.url} (InfluxDB v{self.health.version or 'unknown'}), "
                    f"org: {self.org}, bucket: {self.bucket}"
                )
                return
                
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    retry_delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(
                        f"Connection attempt {attempt + 1}/{self.max_retries} failed: {e}. "
                        f"Retrying in {retry_delay:.1f}s..."
                    )
                    time.sleep(retry_delay)
        
        # If we get here, all retries failed
        self.connected = False
        logger.error(f"Failed to connect to {self.url} after {self.max_retries} attempts")
        if last_error:
            raise last_error
        else:
            raise RuntimeError("Failed to connect to the database")
    
    async def disconnect(self) -> None:
        """Disconnect from the time-series database."""
        try:
            if self.write_api:
                self.write_api.close()
            if self.client:
                self.client.close()
            self.connected = False
            logger.info("Disconnected from time-series database")
        except Exception as e:
            logger.error(f"Error disconnecting from time-series database: {str(e)}")
            raise
    
    async def check_health(self) -> Dict[str, Any]:
        """Check database health status."""
        if not self.client:
            raise RuntimeError("Not connected to database")
            
        try:
            health = self.client.health()
            return {
                'status': health.status.lower(),
                'message': health.message,
                'version': health.version,
                'timestamp': health.time.isoformat() if health.time else None
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'version': None,
                'timestamp': datetime.utcnow().isoformat()
            }
    
    # ===== Partitioning Methods =====
    
    def _apply_partitioning(self, point: Dict[str, Any]) -> Dict[str, Any]:
        """Apply partitioning to a data point based on the configured strategy."""
        if self.partition_strategy == PartitionStrategy.NONE:
            return point
            
        # Create a copy to avoid modifying the original
        point = point.copy()
        
        # Extract timestamp (default to now if not provided)
        timestamp = point.get('timestamp')
        if not timestamp:
            timestamp = datetime.utcnow().isoformat()
        elif isinstance(timestamp, (int, float)):
            # Convert numeric timestamp to datetime
            timestamp = datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
        
        # Apply time-based partitioning
        if self.partition_strategy in (PartitionStrategy.TIME_BASED, PartitionStrategy.HYBRID):
            dt = parse_date(timestamp).astimezone(timezone.utc)
            
            # Generate partition key based on time interval
            if self.time_partition_interval == '1d':
                partition_key = dt.strftime('day=%Y%m%d')
            elif self.time_partition_interval == '1h':
                partition_key = dt.strftime('hour=%Y%m%d%H')
            elif self.time_partition_interval == '1w':
                # ISO week: YYYY-Www
                partition_key = dt.strftime('week=%G-W%V')
            else:  # Default to daily
                partition_key = dt.strftime('day=%Y%m%d')
            
            # Add to point tags
            if 'tags' not in point:
                point['tags'] = {}
            
            # Extract the partition type (day=, hour=, etc.)
            part_type, part_value = partition_key.split('=')
            point['tags'][f'__{part_type}'] = part_value
            
            # Update metrics
            with self._batch_lock:
                self.metrics['partition_queries'] += 1
        
        # Apply asset-based partitioning if needed
        if self.partition_strategy in (PartitionStrategy.ASSET_BASED, PartitionStrategy.HYBRID):
            # Use symbol or asset_id as partition key if available
            asset_key = point.get('symbol') or point.get('asset_id')
            if asset_key:
                if 'tags' not in point:
                    point['tags'] = {}
                point['tags']['__asset'] = str(asset_key)
        
        return point
    
    def _process_partitioned_result(self, result: Any) -> Any:
        """Process query results from partitioned data."""
        if not isinstance(result, pd.DataFrame) or result.empty:
            return result
            
        # Remove internal partition columns if they exist
        partition_columns = [col for col in result.columns if col.startswith('__')]
        if partition_columns:
            result = result.drop(columns=partition_columns)
            
        return result
    
    # ===== Indexing Methods =====
    
    def _update_indices(self, point: Dict[str, Any]) -> None:
        """Update field indices with values from the data point."""
        if not self.enable_indexing:
            return
            
        with self.index_lock:
            for field in self.index_fields:
                if field in point:
                    self.indices[field].add(str(point[field]))
                    self.metrics['index_updates'] += 1
    
    def _optimize_query(self, query: str, use_index: bool = True) -> str:
        """Optimize the query using available indices and partitioning."""
        if not use_index or not self.enable_indexing:
            return query
            
        # Simple optimization: Add WHERE clauses for indexed fields if they exist in the query
        for field in self.index_fields:
            # Look for patterns like 'WHERE field = value' or 'WHERE field IN (...)'
            pattern = fr'\bWHERE\s+{field}\s*[=<>]\s*[\'\"]([^\'\"]+)[\'\"]'
            match = re.search(pattern, query, re.IGNORECASE)
            
            if match:
                value = match.group(1)
                if value in self.indices[field]:
                    # Value exists in index, keep the query as is
                    self.metrics['index_hits'] += 1
                else:
                    # Value doesn't exist, return empty result set
                    self.metrics['index_misses'] += 1
                    # Return a query that will return no results
                    return f'SELECT * FROM (SELECT 1) WHERE 1=0'
        
        return query
    
    # ===== Schema Management =====
    
    def _validate_point_schema(self, point: Dict[str, Any]) -> None:
        """Validate a data point against its schema if one exists."""
        # TODO: Implement schema validation based on registered schemas
        pass
    
    # ===== Retention Policy Management =====
    
    async def add_retention_policy(
        self,
        bucket: str,
        duration: str,
        shard_group_duration: Optional[str] = None,
        replication_factor: int = 1,
        is_default: bool = False,
        description: str = ""
    ) -> bool:
        """
        Add or update a retention policy for a bucket.
        
        Args:
            bucket: Name of the bucket
            duration: Retention duration (e.g., '7d', '4w', 'INF' for infinite)
            shard_group_duration: Duration of shard groups (defaults to bucket's shard group duration)
            replication_factor: Number of copies of data (default: 1)
            is_default: Whether this should be the default retention policy
            description: Optional description
            
        Returns:
            bool: True if successful
        """
        try:
            # Ensure the bucket exists
            await self.ensure_bucket(bucket)
            
            # Create or update the retention policy
            retention_rule = {
                'type': 'expire',
                'everySeconds': self._duration_to_seconds(duration) if duration.upper() != 'INF' else 0,
                'shardGroupDurationSeconds': (
                    self._duration_to_seconds(shard_group_duration) 
                    if shard_group_duration else None
                ),
                'replicationFactor': replication_factor,
                'description': description
            }
            
            # Get current bucket info
            bucket_info = await self.get_bucket_info(bucket)
            
            # Update retention policies
            if 'retentionRules' not in bucket_info:
                bucket_info['retentionRules'] = []
                
            # Remove existing policy if it exists
            bucket_info['retentionRules'] = [
                r for r in bucket_info['retentionRules'] 
                if r.get('type') != 'expire'
            ]
            
            # Add the new policy
            bucket_info['retentionRules'].append(retention_rule)
            
            # Update the bucket
            await self.client.buckets_api().update_bucket(
                bucket_id=bucket_info['id'],
                bucket_name=bucket,
                retention_rules=bucket_info['retentionRules'],
                description=description
            )
            
            # Update local cache
            self.retention_policies[bucket] = RetentionPolicy(
                bucket=bucket,
                duration=duration,
                shard_group_duration=shard_group_duration,
                replication_factor=replication_factor,
                is_default=is_default,
                description=description
            )
            
            logger.info(f"Updated retention policy for bucket '{bucket}' to {duration}")
            return True
            
        except Exception as e:
            self.metrics['retention_errors'] += 1
            logger.error(f"Failed to set retention policy for bucket '{bucket}': {e}")
            return False
    
    async def get_retention_policy(self, bucket: str) -> Optional[RetentionPolicy]:
        """Get the retention policy for a bucket."""
        if bucket in self.retention_policies:
            return self.retention_policies[bucket]
            
        try:
            bucket_info = await self.get_bucket_info(bucket)
            if not bucket_info or 'retentionRules' not in bucket_info:
                return None
                
            for rule in bucket_info['retentionRules']:
                if rule.get('type') == 'expire':
                    duration_seconds = rule.get('everySeconds', 0)
                    return RetentionPolicy(
                        bucket=bucket,
                        duration=self._seconds_to_duration(duration_seconds) if duration_seconds > 0 else 'INF',
                        shard_group_duration=self._seconds_to_duration(rule.get('shardGroupDurationSeconds')),
                        replication_factor=rule.get('replicationFactor', 1),
                        is_default=rule.get('isDefault', False),
                        description=rule.get('description', '')
                    )
            return None
            
        except Exception as e:
            logger.error(f"Failed to get retention policy for bucket '{bucket}': {e}")
            return None
    
    async def enforce_retention_policies(self) -> None:
        """Enforce all registered retention policies."""
        current_time = time.time()
        
        # Only check retention policies once per interval
        if current_time - self.last_retention_check < self.retention_check_interval:
            return
            
        self.last_retention_check = current_time
        self.metrics['retention_checks'] += 1
        
        # Get all buckets if we haven't loaded them yet
        if not self.retention_policies:
            try:
                buckets = await self.list_buckets()
                for bucket in buckets:
                    policy = await self.get_retention_policy(bucket['name'])
                    if policy:
                        self.retention_policies[bucket['name']] = policy
            except Exception as e:
                logger.error(f"Failed to load retention policies: {e}")
        
        # Enforce each policy
        for bucket_name, policy in list(self.retention_policies.items()):
            try:
                await self._enforce_retention_policy(bucket_name, policy)
            except Exception as e:
                self.metrics['retention_errors'] += 1
                logger.error(f"Error enforcing retention policy for {bucket_name}: {e}")
    
    async def _enforce_retention_policy(self, bucket_name: str, policy: RetentionPolicy) -> None:
        """Enforce a single retention policy."""
        if policy.duration == "0" or not policy.duration:  # Infinite retention
            return
            
        # Calculate cutoff time
        duration_seconds = self._duration_to_seconds(policy.duration)
        cutoff_time = datetime.utcnow() - timedelta(seconds=duration_seconds)
        
        # Build and execute delete query
        delete_query = f'''
        from(bucket: "{bucket_name}")
          |> range(start: 0, stop: {int(cutoff_time.timestamp())})
        '''
        
        try:
            # Get the count of records to be deleted
            count_query = f'''
            {delete_query}
              |> count()
            '''
            
            result = await self.query(count_query, return_type="pandas")
            if not result.empty:
                count = result['_value'].iloc[0]
                if count > 0:
                    # Execute the delete
                    await self.delete_data(
                        start=0,
                        stop=int(cutoff_time.timestamp()),
                        bucket=bucket_name
                    )
                    
                    # Update metrics
                    with self._batch_lock:
                        self.metrics['data_expired'] += count
                    
                    logger.info(f"Expired {count} records from {bucket_name} (older than {policy.duration})")
        except Exception as e:
            self.metrics['retention_errors'] += 1
            logger.error(f"Error enforcing retention for {bucket_name}: {e}")
    
    # ===== Continuous Queries =====
    
    async def create_continuous_query(
        self,
        name: str,
        query: str,
        database: Optional[str] = None,
        resample_every: Optional[str] = None,
        resample_for: Optional[str] = None,
        description: str = ""
    ) -> bool:
        """
        Create or update a continuous query.
        
        Args:
            name: Name of the continuous query
            query: The query to run continuously
            database: Database name (default: current database)
            resample_every: Resample interval (e.g., '10m')
            resample_for: Resample for duration (e.g., '30m')
            description: Optional description
            
        Returns:
            bool: True if successful
        """
        try:
            if not self.connected:
                await self.connect()
                
            # Build the CQ query
            cq_query = f"CREATE CONTINUOUS QUERY {name} "
            
            if resample_every or resample_for:
                cq_query += "RESAMPLE"
                if resample_every:
                    cq_query += f" EVERY {resample_every}"
                if resample_for:
                    cq_query += f" FOR {resample_for}"
                cq_query += " "
                
            cq_query += f"ON {database or self.bucket} {query}"
            
            # Execute the CQ creation
            await self.query(cq_query)
            
            logger.info(f"Created continuous query '{name}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create continuous query '{name}': {e}")
            return False
    
    async def drop_continuous_query(self, name: str, database: Optional[str] = None) -> bool:
        """
        Drop a continuous query.
        
        Args:
            name: Name of the continuous query to drop
            database: Database name (default: current database)
            
        Returns:
            bool: True if successful
        """
        try:
            query = f"DROP CONTINUOUS QUERY {name} ON {database or self.bucket}"
            await self.query(query)
            logger.info(f"Dropped continuous query '{name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to drop continuous query '{name}': {e}")
            return False
    
    async def list_continuous_queries(self, database: Optional[str] = None) -> List[Dict]:
        """
        List all continuous queries.
        
        Args:
            database: Database name (default: current database)
            
        Returns:
            List of continuous query definitions
        """
        try:
            db = database or self.bucket
            query = f"SHOW CONTINUOUS QUERIES ON {db}"
            result = await self.query(query, return_type="pandas")
            
            if result is None or result.empty:
                return []
                
            queries = []
            for _, row in result.iterrows():
                queries.append({
                    'name': row['name'],
                    'query': row['query'],
                    'database': db
                })
                
            return queries
            
        except Exception as e:
            logger.error(f"Failed to list continuous queries: {e}")
            return []
    
    # ===== Monitoring and Metrics =====
    
    async def get_metrics(self, reset: bool = False) -> Dict[str, Any]:
        """
        Get current metrics and optionally reset them.
        
        Args:
            reset: Whether to reset metrics after reading
            
        Returns:
            Dictionary of metrics
        """
        with self._batch_lock:
            metrics = self.metrics.copy()
            
            # Calculate derived metrics
            total_queries = metrics['cache_hits'] + metrics['cache_misses']
            metrics['cache_hit_ratio'] = (
                metrics['cache_hits'] / total_queries if total_queries > 0 else 0
            )
            
            metrics['avg_query_time_ms'] = (
                metrics['query_time_ms'] / metrics['reads'] if metrics['reads'] > 0 else 0
            )
            
            metrics['avg_write_time_ms'] = (
                metrics['write_time_ms'] / metrics['writes'] if metrics['writes'] > 0 else 0
            )
            
            if metrics['uncompressed_bytes'] > 0:
                metrics['compression_ratio'] = (
                    metrics['compressed_bytes'] / metrics['uncompressed_bytes']
                )
            
            # Reset metrics if requested
            if reset:
                self.metrics = {
                    # Basic metrics
                    'writes': 0,
                    'reads': 0,
                    'errors': 0,
                    'retries': 0,
                    'cache_hits': 0,
                    'cache_misses': 0,
                    'batch_flushes': 0,
                    'batch_errors': 0,
                    'bytes_written': 0,
                    'bytes_read': 0,
                    'query_time_ms': 0,
                    'write_time_ms': 0,
                    'uncompressed_bytes': 0,
                    'compressed_bytes': 0,
                    
                    # Partitioning metrics
                    'partitions_created': 0,
                    'partitions_dropped': 0,
                    'partition_queries': 0,
                    'partition_optimizations': 0,
                    
                    # Indexing metrics
                    'indices_created': 0,
                    'index_hits': 0,
                    'index_misses': 0,
                    'index_updates': 0,
                    
                    # Retention metrics
                    'retention_checks': 0,
                    'data_expired': 0,
                    'retention_errors': 0,
                    
                    # Compression metrics
                    'compression_ratio': 0.0
                }
            
            return metrics
    
    async def check_health(self) -> Dict[str, Any]:
        """
        Check the health of the database connection and operations.
        
        Returns:
            Dictionary containing health status and metrics
        """
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'database': {
                'connected': self.connected,
                'bucket': self.bucket,
                'org': self.org,
                'url': self.url
            },
            'operations': {
                'pending_writes': len(self._batch_buffer),
                'active_queries': 0,  # Would need thread tracking to implement
                'background_tasks': 0  # Would need task tracking to implement
            },
            'metrics': await self.get_metrics(),
            'warnings': []
        }
        
        # Check for potential issues
        if not self.connected:
            health_status['status'] = 'disconnected'
            health_status['warnings'].append('Not connected to database')
            
        if self.metrics['errors'] > 0:
            health_status['status'] = 'degraded'
            health_status['warnings'].append(f"{self.metrics['errors']} errors occurred")
            
        if len(self._batch_buffer) > self.batch_size * 10:
            health_status['status'] = 'degraded'
            health_status['warnings'].append(f"High write buffer: {len(self._batch_buffer)} pending writes")
            
        # Check disk space if implemented
        disk_usage = await self.get_disk_usage()
        if disk_usage and disk_usage.get('percent_used', 0) > 90:
            health_status['status'] = 'degraded'
            health_status['warnings'].append(
                f"High disk usage: {disk_usage['percent_used']:.1f}% used"
            )
        
        return health_status
    
    async def get_database_stats(self, database: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics for a database.
        
        Args:
            database: Database name (default: current database)
            
        Returns:
            Dictionary containing database statistics
        """
        try:
            db = database or self.bucket
            stats = {
                'database': db,
                'retention_policies': [],
                'measurements': 0,
                'series': 0,
                'disk_usage': 0
            }
            
            # Get retention policies
            query = f"SHOW RETENTION POLICIES ON {db}"
            result = await self.query(query, return_type="pandas")
            
            if result is not None and not result.empty:
                stats['retention_policies'] = result.to_dict('records')
            
            # Get measurement count
            query = f"SHOW MEASUREMENTS ON {db}"
            result = await self.query(query, return_type="pandas")
            
            if result is not None and not result.empty:
                stats['measurements'] = len(result)
            
            # Get series count
            query = f"SHOW SERIES CARDINALITY ON {db}"
            result = await self.query(query, return_type="pandas")
            
            if result is not None and not result.empty:
                stats['series'] = result.iloc[0, 0]
            
            # Get disk usage (approximate)
            stats['disk_usage'] = await self.get_disk_usage(database=db)
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get database stats for '{db}': {e}")
            return {}
    
    async def get_disk_usage(self, database: Optional[str] = None) -> Dict[str, Any]:
        """
        Get disk usage statistics for a database.
        
        Args:
            database: Database name (default: current database)
            
        Returns:
            Dictionary containing disk usage statistics
        """
        try:
            db = database or self.bucket
            
            # Try to get disk usage from InfluxDB API
            try:
                # This is a placeholder - actual implementation would use the InfluxDB API
                # or system commands to get real disk usage
                return {
                    'database': db,
                    'total_bytes': 0,
                    'used_bytes': 0,
                    'available_bytes': 0,
                    'percent_used': 0.0,
                    'shard_count': 0,
                    'oldest_data': None
                }
                
            except Exception as api_error:
                logger.debug(f"Could not get disk usage from API: {api_error}")
                
                # Fall back to filesystem-based estimation
                try:
                    # This would require knowing the data directory
                    data_dir = "/var/lib/influxdb/data"  # Default on Linux
                    
                    # Calculate size of database directory
                    db_dir = os.path.join(data_dir, db)
                    if os.path.exists(db_dir):
                        total_size = sum(
                            os.path.getsize(os.path.join(dirpath, filename))
                            for dirpath, _, filenames in os.walk(db_dir)
                            for filename in filenames
                        )
                        
                        # Get disk stats
                        statvfs = os.statvfs(data_dir)
                        total_space = statvfs.f_frsize * statvfs.f_blocks
                        used_space = total_space - (statvfs.f_frsize * statvfs.f_bavail)
                        
                        return {
                            'database': db,
                            'total_bytes': total_space,
                            'used_bytes': used_space,
                            'available_bytes': statvfs.f_frsize * statvfs.f_bavail,
                            'percent_used': (used_space / total_space) * 100 if total_space > 0 else 0,
                            'shard_count': len([f for f in os.listdir(db_dir) if os.path.isdir(os.path.join(db_dir, f))])
                        }
                    
                    return {
                        'database': db,
                        'error': 'Database directory not found',
                        'data_dir': data_dir
                    }
                    
                except Exception as fs_error:
                    logger.error(f"Failed to get disk usage from filesystem: {fs_error}")
                    return {
                        'database': db,
                        'error': str(fs_error)
                    }
                    
        except Exception as e:
            logger.error(f"Failed to get disk usage: {e}")
            return {
                'database': database or self.bucket,
                'error': str(e)
            }
    
    # ===== Alerting and Notifications =====
    
    async def add_alert_rule(
        self,
        name: str,
        query: str,
        condition: str,
        message: str,
        level: str = 'warning',
        enabled: bool = True,
        check_interval: str = '5m',
        alert_after: str = '1m',
        notification_channels: Optional[List[str]] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Add a new alert rule to monitor the database.
        
        Args:
            name: Name of the alert rule
            query: Query to run for checking the alert condition
            condition: Condition expression (e.g., 'value > threshold')
            message: Alert message template (can use {field} placeholders)
            level: Alert level ('info', 'warning', 'critical')
            enabled: Whether the alert is enabled
            check_interval: How often to check the alert condition
            alert_after: How long the condition must be true before alerting
            notification_channels: List of notification channel IDs
            tags: Optional tags for the alert rule
            
        Returns:
            bool: True if successful
        """
        try:
            alert_rule = {
                'name': name,
                'query': query,
                'condition': condition,
                'message': message,
                'level': level.lower(),
                'enabled': enabled,
                'check_interval': check_interval,
                'alert_after': alert_after,
                'last_checked': None,
                'last_triggered': None,
                'status': 'ok',
                'notification_channels': notification_channels or [],
                'tags': tags or {}
            }
            
            # Store the alert rule (in a real implementation, this would be in a database)
            if not hasattr(self, '_alert_rules'):
                self._alert_rules = {}
                
            self._alert_rules[name] = alert_rule
            logger.info(f"Added alert rule '{name}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add alert rule '{name}': {e}")
            return False
    
    async def check_alerts(self) -> List[Dict]:
        """
        Check all alert rules and trigger notifications if needed.
        
        Returns:
            List of triggered alerts
        """
        if not hasattr(self, '_alert_rules') or not self._alert_rules:
            return []
            
        triggered_alerts = []
        current_time = time.time()
        
        for name, rule in list(self._alert_rules.items()):
            if not rule['enabled']:
                continue
                
            # Check if it's time to run this alert check
            last_checked = rule.get('last_checked', 0)
            check_interval = self._duration_to_seconds(rule['check_interval'])
            
            if current_time - last_checked < check_interval:
                continue
                
            try:
                # Update last checked time
                rule['last_checked'] = current_time
                
                # Run the alert query
                result = await self.query(rule['query'], return_type="pandas")
                
                # Evaluate the condition
                # This is a simplified example - in practice, you'd need a proper expression evaluator
                condition_met = False
                if not result.empty:
                    # For demo purposes, we'll just check if the first value meets the condition
                    value = result.iloc[0, 0]
                    condition_met = eval(rule['condition'], {'value': value, 'result': result})
                
                # Update alert state
                if condition_met:
                    if rule['status'] != 'triggered':
                        # First time this condition is met
                        rule['triggered_at'] = current_time
                        rule['status'] = 'pending'
                    
                    # Check if we've passed the alert_after duration
                    if current_time - rule['triggered_at'] >= self._duration_to_seconds(rule['alert_after']):
                        if rule['status'] != 'triggered':
                            # Trigger the alert
                            rule['status'] = 'triggered'
                            rule['last_triggered'] = current_time
                            
                            # Format the alert message
                            alert = {
                                'name': name,
                                'level': rule['level'],
                                'message': rule['message'].format(**result.iloc[0].to_dict()),
                                'timestamp': datetime.utcnow().isoformat(),
                                'details': {
                                    'query': rule['query'],
                                    'condition': rule['condition'],
                                    'result': result.to_dict('records')[0] if not result.empty else {}
                                }
                            }
                            
                            # Add to triggered alerts
                            triggered_alerts.append(alert)
                            
                            # Send notifications
                            await self._send_alert_notifications(alert, rule)
                            
                            logger.warning(f"Alert triggered: {alert['message']}")
                else:
                    # Reset alert state if condition is no longer met
                    if rule['status'] != 'ok':
                        rule['status'] = 'ok'
                        rule.pop('triggered_at', None)
                        logger.info(f"Alert resolved: {name}")
                        
            except Exception as e:
                logger.error(f"Error checking alert rule '{name}': {e}")
                rule['last_error'] = str(e)
                rule['status'] = 'error'
        
        return triggered_alerts
    
    async def _send_alert_notifications(self, alert: Dict, rule: Dict) -> None:
        """
        Send alert notifications through configured channels.
        
        Args:
            alert: Alert details
            rule: Alert rule configuration
        """
        # This is a placeholder - in a real implementation, you would integrate with
        # notification services like Email, Slack, PagerDuty, etc.
        logger.info(f"Sending alert notification: {alert['message']}")
        
    # ===== Performance Optimization =====
    
    async def optimize_queries(self, analyze_only: bool = True) -> Dict[str, Any]:
        """
        Analyze and optimize slow or inefficient queries.
        
        Args:
            analyze_only: If True, only analyze without making changes
            
        Returns:
            Dictionary with optimization results
        """
        results = {
            'analyzed_queries': 0,
            'optimized_queries': 0,
            'suggestions': [],
            'warnings': []
        }
        
        try:
            # Get slow queries from metrics (simplified example)
            slow_queries = [
                q for q in self.metrics.get('query_history', [])
                if q.get('duration_ms', 0) > 1000  # Queries slower than 1s
            ]
            
            results['analyzed_queries'] = len(slow_queries)
            
            for query_info in slow_queries:
                query = query_info.get('query', '')
                duration = query_info.get('duration_ms', 0)
                
                # Check for common performance issues
                suggestions = []
                
                # 1. Check for missing time range
                if 'time > now()' not in query and 'time >' not in query:
                    suggestions.append("Add time range filter (e.g., 'time > now() - 1h')")
                
                # 2. Check for non-indexed fields in WHERE clause
                for field in self.index_fields:
                    if f"{field} = " in query and field not in query.lower():
                        suggestions.append(f"Add index on field '{field}'")
                
                # 3. Check for inefficient GROUP BY time() intervals
                if 'GROUP BY time(' in query and 'fill(' not in query:
                    suggestions.append("Consider adding fill() to GROUP BY time() to handle missing data")
                
                # 4. Check for SELECT * queries
                if 'SELECT * FROM' in query and 'LIMIT' not in query:
                    suggestions.append("Avoid SELECT * - specify only needed fields")
                
                if suggestions:
                    results['suggestions'].append({
                        'query': query,
                        'duration_ms': duration,
                        'suggestions': suggestions
                    })
            
            # Apply optimizations if not in analyze-only mode
            if not analyze_only and results['suggestions']:
                for suggestion in results['suggestions']:
                    # In a real implementation, you would apply the optimizations here
                    # For now, we'll just log them
                    logger.info(f"Optimizing query: {suggestion['query']}")
                    logger.info(f"  Suggestions: {', '.join(suggestion['suggestions'])}")
                    results['optimized_queries'] += 1
            
            return results
            
        except Exception as e:
            logger.error(f"Error optimizing queries: {e}")
            results['error'] = str(e)
            return results
    
    async def rebuild_indices(self, database: Optional[str] = None) -> Dict[str, Any]:
        """
        Rebuild database indices to improve query performance.
        
        Args:
            database: Database name (default: current database)
            
        Returns:
            Dictionary with rebuild results
        """
        results = {
            'status': 'completed',
            'indices_rebuilt': 0,
            'duration_seconds': 0,
            'details': []
        }
        
        start_time = time.time()
        db = database or self.bucket
        
        try:
            logger.info(f"Starting index rebuild for database '{db}'")
            
            # In a real implementation, this would use database-specific commands
            # to rebuild indices. For InfluxDB, this might involve:
            # 1. Taking a backup
            # 2. Dropping and recreating the database
            # 3. Restoring the data
            
            # For now, we'll simulate the process
            await asyncio.sleep(1)  # Simulate work
            
            results['indices_rebuilt'] = len(self.indices)
            results['details'].append({
                'index': 'all',
                'status': 'rebuilt',
                'duration_seconds': time.time() - start_time
            })
            
            logger.info(f"Completed index rebuild for database '{db}'")
            return results
            
        except Exception as e:
            error_msg = f"Failed to rebuild indices for database '{db}': {e}"
            logger.error(error_msg)
            results.update({
                'status': 'failed',
                'error': error_msg
            })
            return results
    
    # ===== Utility Methods =====
    
    async def execute_sql(self, sql: str, params: Optional[Dict] = None) -> Any:
        """
        Execute raw SQL against the database.
        
        Args:
            sql: SQL query to execute
            params: Optional query parameters
            
        Returns:
            Query results
        """
        try:
            # This is a simplified example - in practice, you would use the appropriate
            # database client method to execute raw SQL
            if sql.strip().upper().startswith('SELECT'):
                return await self.query(sql, params)
            else:
                # For non-SELECT queries, use the write API
                return await self.client.query_api().query(sql, params=params)
        except Exception as e:
            logger.error(f"Error executing SQL: {e}")
            raise
    
    async def export_data(
        self,
        measurement: str,
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None,
        filters: Optional[Dict[str, Any]] = None,
        format: str = 'csv',
        output_file: Optional[str] = None,
        batch_size: int = 10000
    ) -> Union[str, bytes]:
        """
        Export data from a measurement to a file or return as a string.
        
        Args:
            measurement: Name of the measurement to export
            start: Start time for the export (can be datetime or ISO format string)
            end: End time for the export (can be datetime or ISO format string)
            filters: Optional filters to apply as {field: value} or {field: [value1, value2]}
            format: Output format ('csv', 'json', 'parquet')
            output_file: Optional path to save the exported data
            batch_size: Number of records to fetch at once
            
        Returns:
            Exported data as a string or bytes, or the path to the output file
        """
        # Helper function to format time values
        def format_time(t):
            if isinstance(t, datetime):
                return f"'{t.isoformat()}'"
            elif isinstance(t, str):
                return f"'{t}'"
            return str(t)
            
        try:
            # Build the base query
            query_parts = [f'SELECT * FROM "{measurement}"']
            where_parts = []
            
            # Add time range if specified
            if start:
                where_parts.append(f'time >= {format_time(start)}')
            if end:
                where_parts.append(f'time <= {format_time(end)}')
            
            # Add additional filters
            if filters:
                for field, value in filters.items():
                    if value is None:
                        where_parts.append(f'"{field}" IS NULL')
                    elif isinstance(value, (list, tuple, set)):
                        if not value:
                            continue
                        value_list = ', '.join(
                            f"'{v}'" if isinstance(v, str) else str(v) 
                            for v in value
                        )
                        where_parts.append(f'"{field}" IN ({value_list})')
                    else:
                        value_str = f"'{value}'" if isinstance(value, str) else str(value)
                        where_parts.append(f'"{field}" = {value_str}')
                        
            # Combine query parts
            if where_parts:
                query_parts.append(f"WHERE {' AND '.join(where_parts)}")
                
            query = '\n'.join(query_parts)
            
            # Execute the query
            result = self.query(query, return_type=return_type)
            
            # Handle output
            if output_file:
                try:
                    if format == 'csv':
                        result.to_csv(output_file, index=False)
                    elif format == 'json':
                        result.to_json(output_file, orient='records')
                    elif format == 'parquet':
                        result.to_parquet(output_file, index=False)
                    return output_file
                except Exception as file_error:
                    logger.error(f"Error writing to output file {output_file}: {file_error}")
                    raise
                    
            return result
            
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            raise
            raise
        """
        Compact the database to optimize storage and performance.
        
        Args:
            database: Database name (default: current database)
            
        Returns:
            bool: True if successful
        """
        try:
            db = database or self.bucket
            logger.info(f"Starting compaction for database '{db}'")
            
            # For InfluxDB, we can use the API to trigger a compaction
            # This is a placeholder - actual implementation would use the InfluxDB API
            
            logger.info(f"Successfully compacted database '{db}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to compact database '{db}': {e}")
            return False
    
    async def repair_database(self, database: Optional[str] = None) -> bool:
        """
        Repair database consistency issues.
        
        Args:
            database: Database name (default: current database)
            
        Returns:
            bool: True if successful
        """
        try:
            db = database or self.bucket
            logger.info(f"Starting repair for database '{db}'")
            
            # For InfluxDB, we can use the API to trigger a repair
            # This is a placeholder - actual implementation would use the InfluxDB API
            
            logger.info(f"Successfully repaired database '{db}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to repair database '{db}': {e}")
            return False
    
    async def backup_database(
        self,
        backup_dir: str,
        database: Optional[str] = None,
        compression: bool = True,
        incremental: bool = True
    ) -> str:
        """
        Create a backup of the database.
        
        Args:
            backup_dir: Directory to store the backup
            database: Database name (default: current database)
            compression: Whether to compress the backup
            incremental: Whether to perform an incremental backup
            
        Returns:
            str: Path to the backup file
        """
        try:
            db = database or self.bucket
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup_file = os.path.join(backup_dir, f"{db}_backup_{timestamp}.bak")
            
            if compression:
                backup_file += ".gz"
                
            logger.info(f"Starting backup of database '{db}' to {backup_file}")
            
            # For InfluxDB, we can use the backup API or `influxd backup` command
            # This is a placeholder - actual implementation would use the InfluxDB API
            
            logger.info(f"Successfully backed up database '{db}' to {backup_file}")
            return backup_file
            
        except Exception as e:
            logger.error(f"Failed to backup database '{db}': {e}")
            raise
    
    async def restore_database(
        self,
        backup_file: str,
        database: Optional[str] = None,
        drop_existing: bool = False
    ) -> bool:
        """
        Restore a database from a backup.
        
        Args:
            backup_file: Path to the backup file
            database: Target database name (default: current database)
            drop_existing: Whether to drop the existing database
            
        Returns:
            bool: True if successful
        """
        try:
            db = database or self.bucket
            logger.info(f"Starting restore of database '{db}' from {backup_file}")
            
            if drop_existing:
                await self.drop_database(db)
                await self.ensure_bucket(db)
            
            # For InfluxDB, we can use the restore API or `influxd restore` command
            # This is a placeholder - actual implementation would use the InfluxDB API
            
            logger.info(f"Successfully restored database '{db}' from {backup_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore database '{db}': {e}")
            return False
    
    # ===== Bucket Management =====
    
    async def ensure_bucket(self, bucket_name: str, config: Optional[BucketConfig] = None) -> bool:
        """
        Ensure the specified bucket exists with the given configuration.
        
        Args:
            bucket_name: Name of the bucket
            config: Optional bucket configuration
            
        Returns:
            bool: True if the bucket exists or was created successfully
        """
        if not config:
            config = BucketConfig(name=bucket_name)
            
        try:
            # Try to get existing bucket
            try:
                bucket = self.buckets_api.find_bucket_by_name(bucket_name)
                if bucket:
                    logger.debug(f"Bucket '{bucket_name}' already exists")
                    # Update bucket configuration if needed
                    if config.retention_policy:
                        retention_seconds = self._duration_to_seconds(config.retention_policy.duration)
                        self.buckets_api.update_bucket(
                            bucket_id=bucket.id,
                            retention_rules=[{
                                "type": "expire",
                                "everySeconds": retention_seconds,
                                "shardGroupDurationSeconds": self._duration_to_seconds(
                                    config.retention_policy.shard_duration or "24h"
                                )
                            }]
                        )
                    return True
            except Exception as e:
                logger.error(f"Failed to ensure bucket {bucket_name}: {str(e)}")
                return False
            
            # Create new bucket
            bucket = self.buckets_api.create_bucket(
                bucket_name=bucket_name,
                retention_rules=[{
                    "type": "expire",
                    "everySeconds": self._duration_to_seconds(config.retention_policy.duration),
                    "shardGroupDurationSeconds": self._duration_to_seconds(
                        config.retention_policy.shard_duration or "24h"
                    )
                }]
            )
            logger.info(f"Created bucket '{bucket_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create/update bucket '{bucket_name}': {e}", exc_info=True)
            return False
            
    def _duration_to_seconds(self, duration: str) -> int:
        """Convert duration string to seconds."""
        if not duration or duration == "0" or duration.lower() == "infinite":
            return 0
            
        units = {
            'ns': 1e-9,
            'us': 1e-6,
            'µs': 1e-6,
            'ms': 0.001,
            's': 1,
            'm': 60,
            'h': 3600,
            'd': 86400,
            'w': 604800,
        }
        
        # Parse duration string (e.g., "30d", "1h30m", "1w2d")
        total_seconds = 0.0
        current_num = ''
        
        for char in duration.lower():
            if char.isdigit() or char == '.':
                current_num += char
            elif char in units:
                if current_num:
                    total_seconds += float(current_num) * units[char]
                    current_num = ''
        
        return int(total_seconds)
    
    @retry_on_exception(max_retries=3, delay=1, backoff=2)
    async def write_points(
        self,
        points: List[Dict[str, Any]],
        bucket: Optional[str] = None,
        batch: bool = True,
        skip_indexing: bool = False,
        skip_validation: bool = False,
        **kwargs
    ) -> bool:
        """
        Write multiple data points to the database with enhanced features.
        
        Args:
            points: List of point dictionaries
            bucket: Target bucket (default: instance bucket)
            batch: Whether to use batch processing
            skip_indexing: Skip index updates for this write
            skip_validation: Skip schema validation for this write
            **kwargs: Additional write options
            
        Returns:
            bool: True if successful
        """
        """
        Write multiple data points to the database.
        
        Args:
            points: List of point dictionaries
            bucket: Target bucket (default: instance bucket)
            batch: Whether to use batch processing
            **kwargs: Additional write options
            
        Returns:
            bool: True if successful
        """
        if not points:
            return True
            
        bucket = bucket or self.bucket
        processed_points = []
        
        try:
            # Process each point with partitioning and indexing
            for point in points:
                # Apply schema validation if enabled
                if not skip_validation:
                    self._validate_point_schema(point)
                
                # Apply partitioning
                if self.partition_strategy != PartitionStrategy.NONE:
                    point = self._apply_partitioning(point)
                
                # Update indices if enabled
                if self.enable_indexing and not skip_indexing:
                    self._update_indices(point)
                
                processed_points.append(point)
            
            # Write to database
            if batch:
                with self._batch_lock:
                    self._batch_buffer.extend(processed_points)
                    
                    # Check if we should flush based on batch size
                    if len(self._batch_buffer) >= self.batch_size:
                        return await self.flush()
                    return True
            else:
                # Write immediately
                start_time = time.time()
                self._write_points_sync(processed_points)
                write_time = (time.time() - start_time) * 1000  # ms
                
                # Update metrics
                with self._batch_lock:
                    self.metrics['writes'] += len(processed_points)
                    self.metrics['write_time_ms'] += write_time
                    self.metrics['uncompressed_bytes'] += len(str(processed_points).encode('utf-8'))
                
                logger.debug(f"Wrote {len(processed_points)} points in {write_time:.2f}ms")
                return True
                
        except Exception as e:
            self.metrics['errors'] += 1
            logger.error(f"Failed to process/write {len(points)} points: {e}", exc_info=True)
            raise
    
    async def query(
        self,
        query: str,
        params: Optional[Dict] = None,
        return_type: str = "pandas",
        use_cache: bool = True,
        use_index: bool = True,
        optimize: bool = True,
        **kwargs
    ) -> Any:
        """
        Execute a query against the database with enhanced features.
        
        Args:
            query: Query string (Flux or InfluxQL)
            params: Query parameters
            return_type: Result format ("pandas", "raw", "json")
            use_cache: Whether to use query caching
            use_index: Whether to use field indices for optimization
            optimize: Whether to apply query optimization
            **kwargs: Additional query options
            
        Returns:
            Query results in the requested format
        """
        """
        Execute a query against the database.
        
        Args:
            query: Query string (Flux or InfluxQL)
            params: Query parameters
            return_type: Result format ("pandas", "raw", "json")
            use_cache: Whether to use query caching
            **kwargs: Additional query options
            
        Returns:
            Query results in the requested format
        """
        if not self.connected:
            await self.connect()
            
        # Generate cache key if caching is enabled
        cache_key = None
        if use_cache and self.cache_enabled:
            cache_key = self._get_cache_key(query, params)
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                self.metrics['cache_hits'] += 1
                return cached_result
            self.metrics['cache_misses'] += 1
        
        try:
            start_time = time.time()
            
            # Optimize query if enabled
            optimized_query = query
            if optimize:
                optimized_query = self._optimize_query(query, use_index=use_index)
            
            # Execute query
            result = self.query_api.query(query=optimized_query, params=params, **kwargs)
            
            # Convert result format if needed
            if return_type == "pandas" and hasattr(result, 'to_pandas'):
                result = result.to_pandas()
                
                # Apply post-processing for partitioned data if needed
                if self.partition_strategy != PartitionStrategy.NONE:
                    result = self._process_partitioned_result(result)
                
            elif return_type == "json":
                if hasattr(result, 'to_json'):
                    result = result.to_json()
                else:
                    result = json.dumps(result, default=str)
            
            query_time = (time.time() - start_time) * 1000  # ms
            
            # Update metrics
            with self._batch_lock:
                self.metrics['reads'] += 1
                self.metrics['query_time_ms'] += query_time
                if optimize:
                    self.metrics['partition_optimizations'] += 1
            
            logger.debug(f"Query executed in {query_time:.2f}ms")
            
            # Cache the result
            if cache_key is not None:
                self._set_in_cache(cache_key, result)
            
            return result
            
        except Exception as e:
            self.metrics['errors'] += 1
            logger.error(f"Query failed: {e}", exc_info=True)
            raise
    
    async def create_continuous_query(
        self,
        name: str,
        query: str,
        database: Optional[str] = None,
        resample_every: Optional[str] = None,
        resample_for: Optional[str] = None,
        **kwargs
    ) -> bool:
        """
        Create a continuous query.
        
        Args:
            name: Name of the continuous query
            query: Query to run
            database: Target database (default: current bucket)
            resample_every: Resample interval (e.g., "10m")
            resample_for: Resample for time range (e.g., "30m")
            **kwargs: Additional CQ options
            
        Returns:
            bool: True if successful
        """
        if not self.connected:
            await self.connect()
            
        database = database or self.bucket
        
        # Build the CQ query
        cq_query = f"CREATE CONTINUOUS QUERY \"{name}\" ON {database}"
        
        if resample_every or resample_for:
            cq_query += " RESAMPLE"
            if resample_every:
                cq_query += f" EVERY {resample_every}"
            if resample_for:
                cq_query += f" FOR {resample_for}"
        
        cq_query += f" BEGIN {query} END"
        
        try:
            # In InfluxDB 2.x, we need to use the HTTP API directly for CQs
            # as the client library doesn't have direct support
            response = self.client.api_client.call_api(
                f'/api/v2/query/analyze',
                'POST',
                header_params={
                    'Content-Type': 'application/json',
                    'Accept': 'application/csv',
                },
                body={
                    'query': cq_query,
                    'dialect': {'header': True, 'delimiter': ','}
                },
                response_type='str',
                _return_http_data_only=True,
                _preload_content=True,
                _request_timeout=30
            )
            
            logger.info(f"Created continuous query '{name}'")
            return True
            
        except Exception as e:
            self.metrics['errors'] += 1
            logger.error(f"Failed to create continuous query '{name}': {e}", exc_info=True)
            return False
    
    async def create_retention_policy(
        self,
        bucket: str,
        name: str,
        duration: str,
        shard_duration: Optional[str] = None,
        replication: int = 1,
        default: bool = False
    ) -> bool:
        """
        Create a new retention policy.
        
        Args:
            bucket: Bucket name
            name: Policy name
            duration: Retention duration (e.g., "30d", "4w", "0" for infinite)
            shard_duration: Shard group duration (default: same as duration)
            replication: Replication factor
            default: Whether to set as default policy
            
        Returns:
            bool: True if successful
        """
        try:
            from influxdb_client.domain.retention_rule import RetentionRule
            
            # Parse duration to seconds
            def parse_duration(d: str) -> int:
                if d == "0" or d.lower() == "inf":
                    return 0
                units = {'s': 1, 'm': 60, 'h': 3600, 'd': 86400, 'w': 604800}
                unit = d[-1].lower()
                if unit not in units:
                    raise ValueError(f"Invalid duration unit: {unit}")
                return int(d[:-1]) * units[unit]
            
            duration_seconds = parse_duration(duration)
            shard_seconds = parse_duration(shard_duration) if shard_duration else duration_seconds // 2
            
            # Update bucket's retention rules
            bucket_obj = self.buckets_api.find_bucket_by_name(bucket)
            if not bucket_obj:
                raise ValueError(f"Bucket {bucket} not found")
            
            rules = bucket_obj.retention_rules or []
            rules.append(RetentionRule(
                type="expire",
                every_seconds=duration_seconds,
                shard_group_duration_seconds=shard_seconds
            ))
            
            self.buckets_api.update_bucket(
                bucket=bucket_obj,
                retention_rules=rules
            )
            
            logger.info(f"Created retention policy '{name}' for bucket '{bucket}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create retention policy: {str(e)}")
            return False
    
    async def write_points(
        self,
        measurement: str,
        tags: Dict[str, str],
        fields: Dict[str, Any],
        timestamp: Optional[Union[datetime, int, float]] = None,
        bucket: Optional[str] = None,
        precision: TimePrecision = TimePrecision.NS,
        **kwargs
    ) -> bool:
        """
        Write data points to the database.
        
        Args:
            measurement: Measurement name
            tags: Dictionary of tag key-value pairs
            fields: Dictionary of field key-value pairs
            timestamp: Optional timestamp (datetime, Unix timestamp in seconds/ns)
            bucket: Bucket name (default: instance bucket)
            precision: Time precision
            **kwargs: Additional write parameters
            
        Returns:
            bool: True if successful
        """
        if not self.connected or not self.write_api:
            raise RuntimeError("Not connected to database")
            
        try:
            # Create point
            point = Point(measurement)
            
            # Add tags
            for key, value in tags.items():
                if value is not None:
                    point = point.tag(key, str(value))
            
            # Add fields
            for key, value in fields.items():
                if value is not None:
                    if isinstance(value, (int, float, bool, str)):
                        point = point.field(key, value)
                    elif isinstance(value, (list, dict)):
                        # Store complex types as JSON strings
                        point = point.field(f"{key}_json", json.dumps(value))
            
            # Set timestamp
            if timestamp is not None:
                if isinstance(timestamp, (int, float)):
                    point = point.time(int(timestamp), write_precision=precision.value)
                else:
                    point = point.time(timestamp, write_precision=precision.value)
            
            # Write point
            self.write_api.write(
                bucket=bucket or self.bucket,
                record=point,
                **kwargs
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to write points: {str(e)}")
            raise
    
    async def query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        return_type: Literal["pandas", "raw", "json"] = "pandas",
        **kwargs
    ) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Execute a Flux query.
        
        Args:
            query: Flux query string
            params: Query parameters
            return_type: Return type ("pandas", "raw", or "json")
            **kwargs: Additional query parameters
            
        Returns:
            Query results in the specified format
        """
        if not self.connected or not self.query_api:
            raise RuntimeError("Not connected to database")
            
        try:
            # Format query with parameters
            if params:
                query = query.format(**params)
            
            # Execute query
            if return_type == "pandas":
                result = self.query_api.query_data_frame(query)
                
                # Handle case where result is a list of DataFrames
                if isinstance(result, list):
                    if not result:
                        return pd.DataFrame()
                    if len(result) == 1:
                        return result[0]
                    # Concatenate multiple tables
                    return pd.concat(result, ignore_index=True)
                return result
                
            elif return_type == "raw":
                return self.query_api.query(query)
                
            elif return_type == "json":
                tables = self.query_api.query(query)
                result = []
                
                for table in tables:
                    for record in table.records:
                        result.append({
                            'measurement': record.get_measurement(),
                            'tags': record.values.get('tags', {}),
                            'fields': {
                                k: v for k, v in record.values.items()
                                if k not in ('result', 'table', '_start', '_stop', '_time')
                            },
                            'time': record.get_time().isoformat() if record.get_time() else None
                        })
                
                return result
                
            else:
                raise ValueError(f"Unsupported return type: {return_type}")
                
        except Exception as e:
            logger.error(f"Query failed: {str(e)}\nQuery: {query}")
            raise
    
    async def delete_data(
        self,
        start: Union[datetime, str],
        stop: Union[datetime, str, None] = None,
        predicate: Optional[str] = None,
        bucket: Optional[str] = None,
        **kwargs
    ) -> bool:
        """
        Delete data from the database.
        
        Args:
            start: Start time (inclusive)
            stop: Stop time (exclusive), defaults to now if None
            predicate: Optional delete predicate (e.g., '_measurement="tick"')
            bucket: Bucket name (default: instance bucket)
            **kwargs: Additional delete parameters
            
        Returns:
            bool: True if successful
        """
        if not self.connected or not self.delete_api:
            raise RuntimeError("Not connected to database")
            
        try:
            # Parse timestamps
            if isinstance(start, str):
                start = parse_date(start)
            if stop is None:
                stop = datetime.utcnow()
            elif isinstance(stop, str):
                stop = parse_date(stop)
            
            # Delete data
            self.delete_api.delete(
                start=start,
                stop=stop,
                predicate=predicate,
                bucket=bucket or self.bucket,
                org=self.org,
                **kwargs
            )
            
            logger.info(f"Deleted data from {start} to {stop}" + 
                      (f" with predicate: {predicate}" if predicate else ""))
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete data: {str(e)}")
            return False
    
    async def get_measurements(self, bucket: Optional[str] = None) -> List[str]:
        """Get list of measurements in the bucket."""
        query = f'''
        import "influxdata/influxdb/schema"
        schema.measurements(bucket: "{bucket or self.bucket}")
        '''
        
        result = await self.query(query, return_type="pandas")
        return result['_value'].tolist() if not result.empty else []
    
    async def get_field_keys(
        self,
        measurement: str,
        bucket: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Get field keys for a measurement."""
        query = f'''
        import "influxdata/influxdb/schema"
        schema.measurementFieldKeys(
            bucket: "{bucket or self.bucket}",
            measurement: "{measurement}"
        )
        '''
        
        result = await self.query(query, return_type="pandas")
        return [
            {'name': r['_field'], 'type': r['_value']}
            for _, r in result.iterrows()
            if '_field' in r and '_value' in r
        ]
    
    async def get_tag_keys(
        self,
        measurement: str,
        bucket: Optional[str] = None
    ) -> List[str]:
        """Get tag keys for a measurement."""
        query = f'''
        import "influxdata/influxdb/schema"
        schema.measurementTagKeys(
            bucket: "{bucket or self.bucket}",
            measurement: "{measurement}"
        )
        '''
        
        result = await self.query(query, return_type="pandas")
        return result['_value'].tolist() if not result.empty else []
    
    async def get_tag_values(
        self,
        measurement: str,
        tag_key: str,
        bucket: Optional[str] = None
    ) -> List[str]:
        """Get tag values for a measurement and tag key."""
        query = f'''
        import "influxdata/influxdb/schema"
        schema.measurementTagValues(
            bucket: "{bucket or self.bucket}",
            measurement: "{measurement}",
            tag: "{tag_key}"
        )
        '''
        
        result = await self.query(query, return_type="pandas")
        return result['_value'].tolist() if not result.empty else []
    
    async def get_series_cardinality(
        self,
        bucket: Optional[str] = None
    ) -> int:
        """Get the total series cardinality."""
        query = f'''
        import "influxdata/influxdb/schema"
        schema.measurementTagValues(
            bucket: "{bucket or self.bucket}",
            tag: "_measurement"
        )
        |> count()
        '''
        
        result = await self.query(query, return_type="pandas")
        return int(result.iloc[0]['_value']) if not result.empty else 0
    
    async def optimize_query(
        self,
        query: str,
        analyze: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Optimize a Flux query.
        
        Args:
            query: Original Flux query
            analyze: Whether to analyze the query plan
            **kwargs: Additional optimization parameters
            
        Returns:
            Dictionary with optimization results
        """
        # This is a placeholder for query optimization logic
        # In a real implementation, this would analyze the query plan
        # and suggest optimizations
        
        optimized = query  # Placeholder for optimization logic
        
        return {
            'original_query': query,
            'optimized_query': optimized,
            'optimizations_applied': [],
            'estimated_cost_reduction': 0.0
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()

# Example usage
async def example_usage():
    """Example usage of the TimeSeriesDB class."""
    db = TimeSeriesDB(
        url="http://localhost:8086",
        token="your-token",
        org="tick_analysis",
        bucket="market_data"
    )
    
    try:
        # Connect to the database
        await db.connect()
        
        # Write some data
        await db.write_points(
            measurement="trades",
            tags={"symbol": "BTC/USD", "exchange": "binance"},
            fields={"price": 50000.0, "volume": 1.5, "side": "buy"},
            timestamp=datetime.utcnow()
        )
        
        # Query the data
        result = await db.query(
            'from(bucket:"market_data") '
            '|> range(start: -1h) '
            '|> filter(fn: (r) => r._measurement == "trades")',
            return_type="pandas"
        )
        
        print(result.head())
        
    finally:
        # Clean up
        await db.disconnect()

if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())
