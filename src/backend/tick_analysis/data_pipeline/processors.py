"""
Data Processors Module

This module provides data processing components for validation, transformation,
and feature generation in the data pipeline with comprehensive data quality checks.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, AsyncIterator, Tuple, Union, TypeVar
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict
from functools import wraps
import time
import json
from typing_extensions import Protocol

# Type variable for generic data processing
T = TypeVar('T')

logger = logging.getLogger(__name__)

class DataQualityCheck(Enum):
    """Types of data quality checks with severity levels."""
    MISSING_VALUES = (auto(), 'error')
    OUTLIERS = (auto(), 'warning')
    TYPE_VALIDATION = (auto(), 'error')
    RANGE_CHECK = (auto(), 'error')
    TIMESTAMP_ORDER = (auto(), 'warning')
    DUPLICATES = (auto(), 'warning')
    DATA_DRIFT = (auto(), 'critical')
    SCHEMA_VALIDATION = (auto(), 'error')
    LATENCY_CHECK = (auto(), 'warning')
    
    def __new__(cls, value, severity):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.severity = severity
        return obj

class DataQualityMetric(Protocol):
    """Protocol for data quality metrics collection."""
    def record(self, check_type: DataQualityCheck, count: int = 1) -> None: ...
    def get_metrics(self) -> Dict[str, float]: ...

@dataclass
class DataQualityIssue:
    """Represents a data quality issue with context and severity."""
    check_type: DataQualityCheck
    message: str
    field_name: Optional[str] = None
    value: Any = None
    expected: Any = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def severity(self) -> str:
        return self.check_type.severity
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert issue to dictionary for serialization."""
        return {
            'check_type': self.check_type.name,
            'message': self.message,
            'field': self.field_name,
            'value': str(self.value) if self.value is not None else None,
            'expected': str(self.expected) if self.expected is not None else None,
            'timestamp': self.timestamp.isoformat(),
            'severity': self.severity,
            'metadata': self.metadata
        }

class DataProcessor(ABC):
    """Abstract base class for data processors with quality control."""
    
    def __init__(self, metrics_collector: Optional[DataQualityMetric] = None):
        self.metrics = metrics_collector
        self._issues: List[DataQualityIssue] = []
    
    def log_issue(self, issue: DataQualityIssue) -> None:
        """Log a data quality issue and record metrics."""
        self._issues.append(issue)
        if self.metrics:
            self.metrics.record(issue.check_type)
        logger.log(
            logging.ERROR if issue.severity == 'error' else logging.WARNING,
            f"{issue.check_type.name}: {issue.message}"
        )
    
    def get_issues(self) -> List[DataQualityIssue]:
        """Get all recorded data quality issues."""
        return self._issues.copy()
    
    def clear_issues(self) -> None:
        """Clear all recorded issues."""
        self._issues.clear()
    
    @abstractmethod
    async def process(self, data: T) -> T:
        """Process the input data and return the result."""
        pass

class BatchProcessor(DataProcessor):
    """Process data in batches with configurable windowing."""
    
    def __init__(
        self,
        window_size: int = 1000,
        window_slide: Optional[int] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.window_size = window_size
        self.window_slide = window_slide or window_size
        self._buffer: List[Any] = []
    
    async def process_batch(self, batch: List[T]) -> List[T]:
        """Process a single batch of data."""
        raise NotImplementedError
    
    async def process(self, data: T) -> List[T]:
        """Process data using sliding window batching."""
        if not isinstance(data, (list, pd.DataFrame)):
            data = [data]
        
        results = []
        for i in range(0, len(data), self.window_slide):
            window = data[i:i + self.window_size]
            processed = await self.process_batch(window)
            results.extend(processed)
        
        return results

class DataValidator(BatchProcessor):
    """Validates data against schema and quality rules."""
    
    def __init__(
        self,
        schema: Dict[str, type],
        checks: Optional[List[DataQualityCheck]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.schema = schema
        self.checks = checks or list(DataQualityCheck)
    
    async def process_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate a batch of records."""
        validated = []
        for record in batch:
            if self._validate_record(record):
                validated.append(record)
        return validated
    
    def _validate_record(self, record: Dict[str, Any]) -> bool:
        """Validate a single record against schema and rules."""
        is_valid = True
        
        # Schema validation
        for field, expected_type in self.schema.items():
            if field not in record:
                self.log_issue(DataQualityIssue(
                    check_type=DataQualityCheck.SCHEMA_VALIDATION,
                    message=f"Missing required field: {field}",
                    field_name=field,
                    expected=expected_type.__name__
                ))
                is_valid = False
                continue
                
            if not isinstance(record[field], expected_type):
                self.log_issue(DataQualityIssue(
                    check_type=DataQualityCheck.TYPE_VALIDATION,
                    message=f"Type mismatch for field {field}",
                    field_name=field,
                    value=type(record[field]).__name__,
                    expected=expected_type.__name__
                ))
                is_valid = False
        
        # Additional quality checks
        if DataQualityCheck.MISSING_VALUES in self.checks:
            self._check_missing_values(record)
        
        if DataQualityCheck.DUPLICATES in self.checks:
            self._check_duplicates(record)
        
        return is_valid
    
    def _check_missing_values(self, record: Dict[str, Any]) -> None:
        """Check for missing or null values."""
        for field, value in record.items():
            if value is None or (isinstance(value, (str, list, dict)) and not value):
                self.log_issue(DataQualityIssue(
                    check_type=DataQualityCheck.MISSING_VALUES,
                    message=f"Missing or empty value for field: {field}",
                    field_name=field,
                    value=value
                ))
    
    def _check_duplicates(self, record: Dict[str, Any]) -> None:
        """Check for duplicate records."""
        # Implementation depends on how duplicates are identified
        pass

class DataTransformer(BatchProcessor):
    """Transforms data using a series of processing functions."""
    
    def __init__(self, transforms: List[Callable], **kwargs):
        super().__init__(**kwargs)
        self.transforms = transforms
    
    async def process_batch(self, batch: List[T]) -> List[T]:
        """Apply transformation pipeline to batch."""
        result = batch
        for transform in self.transforms:
            if hasattr(transform, '__await__'):
                result = await transform(result)
            else:
                result = transform(result)
        return result

class FeatureGenerator(DataProcessor):
    """Generates features from raw data."""
    
    def __init__(self, feature_definitions: Dict[str, Callable], **kwargs):
        super().__init__(**kwargs)
        self.feature_definitions = feature_definitions
    
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate features from input data."""
        result = data.copy()
        for feature_name, feature_fn in self.feature_definitions.items():
            try:
                if hasattr(feature_fn, '__await__'):
                    result[feature_name] = await feature_fn(data)
                else:
                    result[feature_name] = feature_fn(data)
            except Exception as e:
                self.log_issue(DataQualityIssue(
                    check_type=DataQualityCheck.TYPE_VALIDATION,
                    message=f"Failed to generate feature {feature_name}: {str(e)}",
                    field_name=feature_name,
                    metadata={"error": str(e)}
                ))
        return result

class DataQualityMonitor:
    """Monitors data quality metrics over time."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.issue_counts: Dict[DataQualityCheck, int] = defaultdict(int)
    
    def record(self, check_type: DataQualityCheck, count: int = 1) -> None:
        """Record a data quality issue."""
        self.issue_counts[check_type] += count
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current metrics snapshot."""
        return {
            f"data_quality_{check.name.lower()}": count 
            for check, count in self.issue_counts.items()
        }
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.issue_counts.clear()

def time_processing(metric_name: str):
    """Decorator to measure processing time."""
    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            start_time = time.monotonic()
            try:
                result = await func(self, *args, **kwargs)
                return result
            finally:
                duration = time.monotonic() - start_time
                if hasattr(self, 'metrics') and self.metrics:
                    self.metrics.record_metric(f"{metric_name}_duration_seconds", duration)
        return wrapper
    return decorator

class DataProcessor(ABC):
    """Abstract base class for data processors."""
    
    @abstractmethod
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single data record."""
        pass
    
    @abstractmethod
    async def batch_process(self, data_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of records."""
        pass

class DataValidator(DataProcessor):
    """Validates data quality and integrity."""
    
    def __init__(self, 
                 schema: Dict[str, type] = None,
                 required_fields: List[str] = None,
                 range_checks: Dict[str, Tuple[float, float]] = None,
                 timestamp_field: str = 'timestamp',
                 max_time_gap: float = 300.0):  # 5 minutes in seconds
        """
        Initialize the data validator.
        
        Args:
            schema: Expected field types {field_name: type}
            required_fields: List of required fields
            range_checks: {field_name: (min, max)} for numeric range validation
            timestamp_field: Field containing the timestamp
            max_time_gap: Maximum allowed time gap between records in seconds
        """
        self.schema = schema or {}
        self.required_fields = required_fields or []
        self.range_checks = range_checks or {}
        self.timestamp_field = timestamp_field
        self.max_time_gap = max_time_gap
        self._last_timestamp = None
    
    async def process(self, data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[DataQualityIssue]]:
        """
        Validate a single data record.
        
        Returns:
            Tuple of (validated_data, issues)
        """
        issues = []
        validated = data.copy()
        
        # Check required fields
        for field in self.required_fields:
            if field not in data:
                issues.append(DataQualityIssue(
                    check_type=DataQualityCheck.MISSING_VALUES,
                    message=f"Missing required field: {field}",
                    field_name=field,
                    severity='ERROR'
                ))
        
        # Type validation
        for field, expected_type in self.schema.items():
            if field in data and not isinstance(data[field], expected_type):
                actual_type = type(data[field]).__name__
                issues.append(DataQualityIssue(
                    check_type=DataQualityCheck.TYPE_VALIDATION,
                    message=f"Type mismatch for {field}: expected {expected_type.__name__}, got {actual_type}",
                    field_name=field,
                    value=data[field],
                    expected=expected_type.__name__,
                    severity='ERROR'
                ))
        
        # Range checks
        for field, (min_val, max_val) in self.range_checks.items():
            if field in data and isinstance(data[field], (int, float)):
                if not (min_val <= data[field] <= max_val):
                    issues.append(DataQualityIssue(
                        check_type=DataQualityCheck.RANGE_CHECK,
                        message=f"Value {data[field]} for {field} is outside allowed range [{min_val}, {max_val}]",
                        field_name=field,
                        value=data[field],
                        expected=f"[{min_val}, {max_val}]",
                        severity='WARNING'
                    ))
        
        # Timestamp validation
        if self.timestamp_field in data:
            try:
                if isinstance(data[self.timestamp_field], (int, float)):
                    timestamp = float(data[self.timestamp_field])
                    if self._last_timestamp is not None:
                        time_gap = timestamp - self._last_timestamp
                        if time_gap > self.max_time_gap:
                            issues.append(DataQualityIssue(
                                check_type=DataQualityCheck.TIMESTAMP_ORDER,
                                message=f"Large time gap detected: {time_gap:.2f}s",
                                field_name=self.timestamp_field,
                                value=timestamp,
                                expected=f"< {self.max_time_gap}s gap",
                                severity='WARNING'
                            ))
                    self._last_timestamp = timestamp
                
                # Convert to datetime if it's a string
                if isinstance(data[self.timestamp_field], str):
                    try:
                        validated[self.timestamp_field] = pd.to_datetime(data[self.timestamp_field])
                    except (ValueError, TypeError) as e:
                        issues.append(DataQualityIssue(
                            check_type=DataQualityCheck.TYPE_VALIDATION,
                            message=f"Invalid timestamp format: {data[self.timestamp_field]}",
                            field_name=self.timestamp_field,
                            value=data[self.timestamp_field],
                            severity='ERROR'
                        ))
                        
            except Exception as e:
                issues.append(DataQualityIssue(
                    check_type=DataQualityCheck.TYPE_VALIDATION,
                    message=f"Error processing timestamp: {str(e)}",
                    field_name=self.timestamp_field,
                    value=data.get(self.timestamp_field),
                    severity='ERROR'
                ))
        
        return validated, issues
    
    async def batch_process(self, data_batch: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[DataQualityIssue]]:
        """
        Validate a batch of records.
        
        Returns:
            Tuple of (validated_records, issues)
        """
        validated_batch = []
        all_issues = []
        
        for record in data_batch:
            validated, issues = await self.process(record)
            if validated is not None:
                validated_batch.append(validated)
            all_issues.extend(issues)
        
        return validated_batch, all_issues

class FeatureGenerator(DataProcessor):
    """Generates features from raw market data."""
    
    def __init__(self, 
                 timestamp_field: str = 'timestamp',
                 price_field: str = 'price',
                 volume_field: str = 'volume',
                 window_sizes: List[int] = None):
        """
        Initialize the feature generator.
        
        Args:
            timestamp_field: Field containing the timestamp
            price_field: Field containing the price
            volume_field: Field containing the volume
            window_sizes: List of window sizes for rolling features
        """
        self.timestamp_field = timestamp_field
        self.price_field = price_field
        self.volume_field = volume_field
        self.window_sizes = window_sizes or [5, 10, 20, 50, 100]
        self._price_history = {}
        self._volume_history = {}
    
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate features for a single data record.
        
        Returns:
            Dictionary with original data and generated features
        """
        if not data or self.price_field not in data:
            return data
        
        symbol = data.get('symbol', 'default')
        timestamp = data.get(self.timestamp_field)
        price = float(data.get(self.price_field, 0))
        volume = float(data.get(self.volume_field, 0))
        
        # Initialize history for this symbol if needed
        if symbol not in self._price_history:
            self._price_history[symbol] = []
            self._volume_history[symbol] = []
        
        # Add current data to history
        self._price_history[symbol].append((timestamp, price))
        self._volume_history[symbol].append((timestamp, volume))
        
        # Keep only the most recent data (up to the largest window size)
        max_window = max(self.window_sizes) if self.window_sizes else 100
        self._price_history[symbol] = self._price_history[symbol][-max_window*2:]
        self._volume_history[symbol] = self._volume_history[symbol][-max_window*2:]
        
        # Extract timestamps and values
        timestamps, prices = zip(*self._price_history[symbol]) if self._price_history[symbol] else ([], [])
        _, volumes = zip(*self._volume_history[symbol]) if self._volume_history[symbol] else ([], [])
        
        # Create features dictionary
        features = {}
        
        # Basic features
        features['price'] = price
        features['log_return'] = self._calculate_log_return(prices)
        features['volume'] = volume
        features['volume_ratio'] = self._calculate_volume_ratio(volumes)
        
        # Rolling features for different window sizes
        for window in self.window_sizes:
            if len(prices) >= window:
                window_prices = prices[-window:]
                window_volumes = volumes[-window:] if volumes else []
                
                # Price features
                features[f'sma_{window}'] = sum(window_prices) / len(window_prices)
                features[f'std_{window}'] = np.std(window_prices)
                features[f'max_{window}'] = max(window_prices)
                features[f'min_{window}'] = min(window_prices)
                
                # Volume features
                if window_volumes:
                    features[f'volume_ma_{window}'] = sum(window_volumes) / len(window_volumes)
        
        # Add features to the original data
        result = data.copy()
        result.update(features)
        
        return result
    
    async def batch_process(self, data_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate features for a batch of records."""
        return [await self.process(record) for record in data_batch]
    
    def _calculate_log_return(self, prices: List[float]) -> float:
        """Calculate log return from price series."""
        if len(prices) < 2:
            return 0.0
        return np.log(prices[-1] / prices[-2]) if prices[-2] != 0 else 0.0
    
    def _calculate_volume_ratio(self, volumes: List[float]) -> float:
        """Calculate volume ratio compared to previous period."""
        if len(volumes) < 2 or sum(volumes[:-1]) == 0:
            return 1.0
        return volumes[-1] / (sum(volumes[:-1]) / (len(volumes) - 1))
