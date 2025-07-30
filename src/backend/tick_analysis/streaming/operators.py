"""
Stream processing operators for drift detection.

This module provides various stream processing operators that can be composed
to build complex data processing pipelines with drift detection capabilities.
"""

from typing import Any, Callable, Dict, List, Optional, Union, TypeVar, Generic, Tuple, Type, Set
from dataclasses import dataclass, field
import asyncio
import time
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
import pandas as pd
from enum import Enum

from tick_analysis.exceptions import StreamProcessingError, ConfigurationError
from tick_analysis.monitoring.drift_monitor import DriftMonitor, DriftAlert

logger = logging.getLogger(__name__)

T = TypeVar('T')

class StreamOperator(Generic[T]):
    """Base class for all stream operators."""
    
    def process(self, item: T) -> T:
        """Process a single item."""
        raise NotImplementedError
    
    async def process_async(self, item: T) -> T:
        """Process a single item asynchronously."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.process, item
        )
    
    def batch_process(self, items: List[T]) -> List[T]:
        """Process a batch of items."""
        return [self.process(item) for item in items]
    
    async def batch_process_async(self, items: List[T]) -> List[T]:
        """Process a batch of items asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.batch_process, items
        )
    
    def __or__(self, other: 'StreamOperator') -> 'StreamPipeline':
        """Compose operators using the pipe operator."""
        return StreamPipeline([self, other])


@dataclass
class MapOperator(StreamOperator[T]):
    """Applies a function to each item in the stream."""
    
    func: Callable[[T], T]
    
    def process(self, item: T) -> T:
        try:
            return self.func(item)
        except Exception as e:
            raise StreamProcessingError(f"Error in MapOperator: {str(e)}") from e


@dataclass
class FilterOperator(StreamOperator[T]):
    """Filters items based on a predicate function."""
    
    predicate: Callable[[T], bool]
    
    def process(self, item: T) -> Optional[T]:
        try:
            return item if self.predicate(item) else None
        except Exception as e:
            raise StreamProcessingError(f"Error in FilterOperator: {str(e)}") from e


@dataclass
class BatchOperator(StreamOperator[T]):
    """Batches items into fixed-size batches."""
    
    batch_size: int
    timeout_seconds: float = 0.1
    _buffer: List[T] = None
    
    def __post_init__(self):
        self._buffer = []
        self._last_flush = time.time()
    
    def process(self, item: T) -> Optional[List[T]]:
        self._buffer.append(item)
        current_time = time.time()
        
        if (len(self._buffer) >= self.batch_size or 
            (current_time - self._last_flush) >= self.timeout_seconds):
            return self._flush()
        return None
    
    def _flush(self) -> List[T]:
        """Flush the current buffer."""
        if not self._buffer:
            return None
            
        batch = self._buffer.copy()
        self._buffer.clear()
        self._last_flush = time.time()
        return batch
    
    def flush(self) -> Optional[List[T]]:
        """Force flush the buffer."""
        return self._flush()


@dataclass
class ParallelMapOperator(StreamOperator[T]):
    """Applies a function to items in parallel."""
    
    func: Callable[[T], T]
    max_workers: int = 4
    use_processes: bool = False
    
    def __post_init__(self):
        executor_cls = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        self._executor = executor_cls(max_workers=self.max_workers)
    
    def process(self, item: T) -> T:
        try:
            return self.func(item)
        except Exception as e:
            raise StreamProcessingError(f"Error in ParallelMapOperator: {str(e)}") from e
    
    async def process_async(self, item: T) -> T:
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(
                self._executor, self.process, item
            )
        except Exception as e:
            raise StreamProcessingError(f"Error in async processing: {str(e)}") from e
    
    def batch_process(self, items: List[T]) -> List[T]:
        try:
            return list(self._executor.map(self.func, items))
        except Exception as e:
            raise StreamProcessingError(f"Error in batch processing: {str(e)}") from e
    
    async def batch_process_async(self, items: List[T]) -> List[T]:
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(
                self._executor, self.batch_process, items
            )
        except Exception as e:
            raise StreamProcessingError(f"Error in async batch processing: {str(e)}") from e
    
    def __del__(self):
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=True)


@dataclass
class DriftDetectorOperator(StreamOperator[Dict[str, Any]]):
    """Detects drift in the data stream."""
    
    monitor: DriftMonitor
    features: List[str]
    batch_mode: bool = True
    
    def process(self, item: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Update monitor with current item
            self.monitor.update(item)
            
            # Get latest drift results
            if self.batch_mode:
                # Only check for drift when the window is full
                if len(self.monitor.buffer) >= self.monitor.window_size:
                    drift_results = self.monitor.get_latest_drift_results()
                    item['_drift'] = {
                        'is_drifted': any(r.is_drifted for r in drift_results),
                        'results': [r.to_dict() for r in drift_results]
                    }
            else:
                # Check for drift on every item (less efficient)
                drift_results = self.monitor.get_latest_drift_results()
                item['_drift'] = {
                    'is_drifted': any(r.is_drifted for r in drift_results),
                    'results': [r.to_dict() for r in drift_results]
                }
            
            return item
        except Exception as e:
            raise StreamProcessingError(f"Error in DriftDetectorOperator: {str(e)}") from e


@dataclass
class AlertOperator(StreamOperator[Dict[str, Any]]):
    """Processes alerts from the stream."""
    
    monitor: DriftMonitor
    alert_handlers: List[Callable[[DriftAlert], None]]
    
    def process(self, item: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Check for new alerts
            alerts = self.monitor.get_new_alerts()
            
            # Process each alert
            for alert in alerts:
                for handler in self.alert_handlers:
                    try:
                        handler(alert)
                    except Exception as e:
                        logger.error(f"Error in alert handler: {str(e)}", exc_info=True)
            
            # Add alerts to the item
            if alerts:
                item['_alerts'] = [{
                    'severity': alert.severity.value,
                    'message': alert.message,
                    'timestamp': alert.timestamp.isoformat(),
                    'metadata': getattr(alert, 'metadata', {})
                } for alert in alerts]
            
            return item
        except Exception as e:
            raise StreamProcessingError(f"Error in AlertOperator: {str(e)}") from e


class WindowType(Enum):
    """Types of windows for windowing operations."""
    TUMBLING = "tumbling"
    SLIDING = "sliding"
    SESSION = "session"


class WindowOperator(StreamOperator[T]):
    """Windowing operator that groups items into windows."""
    
    def __init__(
        self,
        window_size: int,
        slide_size: Optional[int] = None,
        window_type: WindowType = WindowType.TUMBLING,
        timestamp_extractor: Optional[Callable[[T], float]] = None,
        key_extractor: Optional[Callable[[T], Any]] = None
    ):
        """Initialize the window operator.
        
        Args:
            window_size: Size of the window in number of items or seconds
            slide_size: Slide interval (defaults to window_size for tumbling windows)
            window_type: Type of window (tumbling, sliding, session)
            timestamp_extractor: Function to extract timestamp from items
            key_extractor: Function to extract key for keyed windows
        """
        self.window_size = window_size
        self.slide_size = slide_size if slide_size is not None else window_size
        self.window_type = window_type
        self.timestamp_extractor = timestamp_extractor or (lambda x: time.time())
        self.key_extractor = key_extractor or (lambda x: None)
        
        # State
        self.windows = defaultdict(deque)  # key -> deque of (timestamp, item)
        self.last_processed = defaultdict(float)
        self.session_timeout = window_size if window_type == WindowType.SESSION else None
    
    def process(self, item: T) -> List[T]:
        """Process an item and return completed windows."""
        key = self.key_extractor(item)
        timestamp = self.timestamp_extractor(item)
        
        if self.window_type == WindowType.TUMBLING:
            return self._process_tumbling_window(key, timestamp, item)
        elif self.window_type == WindowType.SLIDING:
            return self._process_sliding_window(key, timestamp, item)
        elif self.window_type == WindowType.SESSION:
            return self._process_session_window(key, timestamp, item)
        else:
            raise ValueError(f"Unknown window type: {self.window_type}")
    
    def _process_tumbling_window(self, key: Any, timestamp: float, item: T) -> List[T]:
        """Process item for tumbling windows."""
        window_id = timestamp // self.window_size
        window_key = (key, window_id)
        
        # Add item to window
        self.windows[window_key].append((timestamp, item))
        
        # Check if window is complete
        if len(self.windows[window_key]) >= self.window_size:
            window_items = [item for _, item in self.windows.pop(window_key)]
            return window_items
        
        return []
    
    def _process_sliding_window(self, key: Any, timestamp: float, item: T) -> List[List[T]]:
        """Process item for sliding windows."""
        window_start = timestamp - self.window_size
        window_id = timestamp // self.slide_size
        window_key = (key, window_id)
        
        # Add item to window
        self.windows[window_key].append((timestamp, item))
        
        # Remove old items
        while self.windows[window_key] and self.windows[window_key][0][0] < window_start:
            self.windows[window_key].popleft()
        
        # Return current window if it has items
        if self.windows[window_key]:
            return [[item for _, item in self.windows[window_key]]]
        
        return []
    
    def _process_session_window(self, key: Any, timestamp: float, item: T) -> List[T]:
        """Process item for session windows."""
        if key not in self.windows or not self.windows[key]:
            # Start new session
            self.windows[key] = deque([(timestamp, item)])
            self.last_processed[key] = timestamp
            return []
        
        # Add to existing session
        self.windows[key].append((timestamp, item))
        self.last_processed[key] = timestamp
        
        # Check if session has timed out
        if timestamp - self.last_processed[key] > self.session_timeout:
            session_items = [item for _, item in self.windows.pop(key)]
            return session_items
        
        return []


class AggregateOperator(StreamOperator[List[T]]):
    """Aggregates items in a window."""
    
    def __init__(
        self,
        aggregations: Dict[str, Tuple[Callable, Callable]],
        key_selector: Optional[Callable[[T], Any]] = None
    ):
        """Initialize the aggregation operator.
        
        Args:
            aggregations: Dictionary of field name to (initializer, aggregator) pairs
            key_selector: Function to extract key for grouping (None for global aggregation)
        """
        self.aggregations = aggregations
        self.key_selector = key_selector
        self.state = {}
    
    def process(self, items: List[T]) -> List[Dict[str, Any]]:
        """Process a batch of items and return aggregated results."""
        if not items:
            return []
            
        if self.key_selector:
            # Group by key
            groups = {}
            for item in items:
                key = self.key_selector(item)
                if key not in groups:
                    groups[key] = []
                groups[key].append(item)
            
            # Aggregate each group
            return [self._aggregate_group(key, group_items) for key, group_items in groups.items()]
        else:
            # Global aggregation
            return [self._aggregate_group(None, items)]
    
    def _aggregate_group(self, key: Any, items: List[T]) -> Dict[str, Any]:
        """Aggregate a group of items."""
        # Initialize state for this key if needed
        if key not in self.state:
            self.state[key] = {
                field: initializer() 
                for field, (initializer, _) in self.aggregations.items()
            }
        
        # Apply aggregations
        result = {'key': key} if key is not None else {}
        
        for field, (_, aggregator) in self.aggregations.items():
            values = [getattr(item, field, None) for item in items if hasattr(item, field)]
            if values:
                self.state[key][field] = aggregator(self.state[key][field], values)
                result[field] = self.state[key][field]
        
        return result


class JoinOperator(StreamOperator[Dict[str, Any]]):
    """Joins two streams based on a key."""
    
    def __init__(
        self,
        left_key: Callable[[Any], Any],
        right_key: Callable[[Any], Any],
        join_type: str = 'inner',
        window_size: int = 1000,
        expire_after: Optional[float] = None
    ):
        """Initialize the join operator.
        
        Args:
            left_key: Function to extract join key from left stream
            right_key: Function to extract join key from right stream
            join_type: Type of join ('inner', 'left', 'right', 'outer')
            window_size: Maximum number of items to keep in each join window
            expire_after: Seconds after which to expire entries (None for no expiration)
        """
        self.left_key = left_key
        self.right_key = right_key
        self.join_type = join_type
        self.window_size = window_size
        self.expire_after = expire_after
        
        # State
        self.left_window = {}
        self.right_window = {}
        self.left_timestamps = {}
        self.right_timestamps = {}
    
    def process(self, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process an item from either stream."""
        stream = item.get('_stream', 'left')
        
        if stream == 'left':
            return self._process_left(item)
        else:
            return self._process_right(item)
    
    def _process_left(self, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process an item from the left stream."""
        key = self.left_key(item)
        timestamp = time.time()
        
        # Add to window
        if key not in self.left_window:
            self.left_window[key] = []
        
        self.left_window[key].append(item)
        self.left_timestamps[key] = timestamp
        
        # Enforce window size
        if len(self.left_window[key]) > self.window_size:
            self.left_window[key].pop(0)
        
        # Join with right window
        return self._join('left', key, item)
    
    def _process_right(self, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process an item from the right stream."""
        key = self.right_key(item)
        timestamp = time.time()
        
        # Add to window
        if key not in self.right_window:
            self.right_window[key] = []
        
        self.right_window[key].append(item)
        self.right_timestamps[key] = timestamp
        
        # Enforce window size
        if len(self.right_window[key]) > self.window_size:
            self.right_window[key].pop(0)
        
        # Join with left window
        return self._join('right', key, item)
    
    def _join(self, side: str, key: Any, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Perform the actual join operation."""
        results = []
        
        if side == 'left':
            # Left stream item joining with right window
            if key in self.right_window:
                for right_item in self.right_window[key]:
                    results.append({**item, **right_item})
            elif self.join_type in ['left', 'outer']:
                results.append({**item, **{k: None for k in self.right_window.get(next(iter(self.right_window), {}), {})}})
        else:
            # Right stream item joining with left window
            if key in self.left_window:
                for left_item in self.left_window[key]:
                    results.append({**left_item, **item})
            elif self.join_type in ['right', 'outer']:
                results.append({**{k: None for k in self.left_window.get(next(iter(self.left_window), {}), {})}, **item})
        
        # Clean up expired entries
        self._cleanup()
        
        return results
    
    def _cleanup(self) -> None:
        """Clean up expired entries."""
        if self.expire_after is None:
            return
            
        current_time = time.time()
        
        # Clean left window
        expired_keys = [
            k for k, ts in self.left_timestamps.items() 
            if current_time - ts > self.expire_after
        ]
        for k in expired_keys:
            self.left_window.pop(k, None)
            self.left_timestamps.pop(k, None)
        
        # Clean right window
        expired_keys = [
            k for k, ts in self.right_timestamps.items() 
            if current_time - ts > self.expire_after
        ]
        for k in expired_keys:
            self.right_window.pop(k, None)
            self.right_timestamps.pop(k, None)


class StreamPipeline(StreamOperator[T]):
    """A pipeline of stream operators with backpressure support and monitoring.
    
    The pipeline manages the flow of data through a series of operators, applying
    backpressure when the system is under heavy load to prevent out-of-memory errors
    and ensure stable operation.
    
    Attributes:
        operators: List of operators in the pipeline
        max_queue_size: Maximum number of items allowed in the processing queue
        backpressure_threshold: Queue size threshold (0-1) at which to apply backpressure
        backoff_factor: Multiplier for exponential backoff
        min_backoff: Minimum backoff time in seconds
        max_backoff: Maximum backoff time in seconds
        metrics_interval: How often to collect metrics (in seconds)
        last_metrics_time: Timestamp of last metrics collection
    """
    
    def __init__(
        self, 
        operators: List[StreamOperator] = None,
        max_queue_size: int = 1000,
        backpressure_threshold: float = 0.8,
        backoff_factor: float = 1.5,
        min_backoff: float = 0.1,
        max_backoff: float = 5.0,
        metrics_interval: float = 5.0
    ):
        """Initialize the stream pipeline.
        
        Args:
            operators: List of stream operators in processing order
            max_queue_size: Maximum number of items in the processing queue
            backpressure_threshold: Queue threshold (0-1) for applying backpressure
            backoff_factor: Multiplier for exponential backoff
            min_backoff: Minimum backoff time in seconds
            max_backoff: Maximum backoff time in seconds
            metrics_interval: How often to collect metrics (in seconds)
        """
        super().__init__(name="StreamPipeline")
        self.operators = operators or []
        self.max_queue_size = max_queue_size
        self.backpressure_threshold = backpressure_threshold
        self.backoff_factor = backoff_factor
        self.min_backoff = min_backoff
        self.max_backoff = max_backoff
        self.current_backoff = min_backoff
        self.metrics_interval = metrics_interval
        self.last_metrics_time = time.time()
        
        # Performance metrics
        self.metrics = {
            'total_processed': 0,
            'total_errors': 0,
            'avg_processing_time': 0.0,
            'max_processing_time': 0.0,
            'min_processing_time': float('inf'),
            'queue_size': 0,
            'throughput': 0.0,  # items per second
            'last_processed': None,
            'operator_metrics': {}
        }
        self._processing_start = None
        self._last_processed_count = 0
        self._last_processed_time = time.time()
    
    def process(self, item: T) -> T:
        """Process an item through the pipeline."""
        try:
            result = item
            for op in self.operators:
                result = op.process(result)
                if result is None:  # Filtered out
                    return None
            return result
        except Exception as e:
            raise StreamProcessingError(f"Error in stream pipeline: {str(e)}") from e
    
    async def process_async(self, item: T) -> T:
        """Process an item asynchronously through the pipeline."""
        try:
            result = item
            for op in self.operators:
                if hasattr(op, 'process_async'):
                    result = await op.process_async(result)
                else:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, op.process, result
                    )
                if result is None:  # Filtered out
                    return None
            return result
        except Exception as e:
            raise StreamProcessingError(f"Error in async stream pipeline: {str(e)}") from e
    
    def batch_process(self, items: List[T], max_workers: int = None) -> List[T]:
        """Process a batch of items through the pipeline with parallel processing.
        
        Args:
            items: List of items to process
            max_workers: Maximum number of worker threads to use (default: min(32, CPU count * 5))
            
        Returns:
            List of processed items (excluding any that were filtered out or failed)
        """
        if not items:
            return []
            
        # For small batches, use sequential processing to avoid thread overhead
        if len(items) <= 10:
            results = []
            for item in items:
                try:
                    result = self.process(item)
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Error processing batch item: {e}")
                    continue
            return results
            
        # For larger batches, use thread pool for parallel processing
        import concurrent.futures
        from functools import partial
        
        if max_workers is None:
            max_workers = min(32, (os.cpu_count() or 1) * 5)
        
        processed_count = 0
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all items for processing
            future_to_item = {
                executor.submit(self.process, item): item
                for item in items
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                    processed_count += 1
                    
                    # Log progress for large batches
                    if processed_count % 100 == 0:
                        logger.info(
                            f"Processed {processed_count}/{len(items)} items "
                            f"({processed_count/len(items):.1%})"
                        )
                        
                except Exception as e:
                    logger.error(f"Error processing batch item: {e}")
        
        return results
    
    async def batch_process_async(
        self, 
        items: List[T], 
        max_concurrent: int = None,
        chunk_size: int = 100
    ) -> List[T]:
        """Process a batch of items asynchronously with controlled concurrency.
        
        Args:
            items: List of items to process
            max_concurrent: Maximum number of concurrent operations (default: min(100, CPU count * 10))
            chunk_size: Number of items to process in each chunk (for progress reporting)
            
        Returns:
            List of processed items (excluding any that were filtered out or failed)
        """
        if not items:
            return []
            
        # For small batches, use sequential processing to avoid overhead
        if len(items) <= 5:
            results = []
            for item in items:
                try:
                    result = await self.process_async(item)
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Error processing async batch item: {e}")
                    continue
            return results
        
        # Configure concurrency
        if max_concurrent is None:
            max_concurrent = min(100, (os.cpu_count() or 1) * 10)
        
        semaphore = asyncio.Semaphore(max_concurrent)
        results = []
        processed_count = 0
        
        async def process_with_semaphore(item):
            async with semaphore:
                try:
                    return await self.process_async(item)
                except Exception as e:
                    logger.error(f"Error in async batch processing: {e}")
                    return None
        
        # Process in chunks to avoid creating too many tasks at once
        for i in range(0, len(items), chunk_size):
            chunk = items[i:i + chunk_size]
            
            # Process current chunk
            chunk_tasks = [process_with_semaphore(item) for item in chunk]
            chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)
            
            # Collect successful results
            for result in chunk_results:
                if result is not None and not isinstance(result, Exception):
                    results.append(result)
            
            # Update progress
            processed_count += len(chunk)
            logger.info(
                f"Processed {processed_count}/{len(items)} items "
                f"({processed_count/len(items):.1%})"
            )
        
        return results
    
    def __or__(self, other: 'StreamOperator') -> 'StreamPipeline':
        """Compose with another operator or pipeline."""
        if isinstance(other, StreamPipeline):
            return StreamPipeline(self.operators + other.operators)
        return StreamPipeline(self.operators + [other])


# Example usage:
if __name__ == "__main__":
    # Create a simple pipeline
    pipeline = (
        MapOperator(lambda x: {**x, 'processed': True}) |
        FilterOperator(lambda x: x.get('value', 0) > 10) |
        BatchOperator(batch_size=10)
    )
    
    # Process some data
    data = [{'value': i, 'id': i} for i in range(20)]
    
    # Process in batches
    for item in data:
        result = pipeline.process(item)
        if result:
            print(f"Batch result: {result}")
