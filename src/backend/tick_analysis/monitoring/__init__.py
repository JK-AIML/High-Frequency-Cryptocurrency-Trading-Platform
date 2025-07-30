"""
Monitoring and Drift Detection

This package provides tools for monitoring data quality and detecting drift
in both batch and streaming data processing pipelines.

Main Components:
- UnifiedDriftDetector: Core class for drift detection
- DriftMonitor: Real-time monitoring with alerting
- AlertRule: Define conditions for triggering alerts
- Legacy adapters for backward compatibility

Example:
    >>> from tick_analysis.monitoring import UnifiedDriftDetector, DriftType, DetectionMethod
    >>> import pandas as pd
    >>> import numpy as np
    
    # Generate example data
    >>> reference_data = pd.DataFrame({
    ...     'feature1': np.random.normal(0, 1, 1000),
    ...     'feature2': np.random.normal(10, 2, 1000)
    ... })
    >>> 
    >>> # Initialize detector
    >>> detector = UnifiedDriftDetector(reference_data)
    >>> 
    >>> # Detect drift
    >>> current_data = pd.DataFrame({
"""

from .unified_drift import (
    UnifiedDriftDetector,
    DetectionMethod,
    DriftType,
    DriftResult
)

from .drift_monitor import (
    DriftMonitor,
    AlertRule,
    AlertSeverity,
    DriftAlert
)

from .legacy_adapter import (
    DataDriftDetector as LegacyDataDriftDetector,
    DriftDetector as LegacyDriftDetector
)

# Re-export for backward compatibility
from .drift_detector import DriftDetector, DriftMetric
from tick_analysis.data_pipeline.drift_detection import DataDriftDetector

__all__ = [
    # Unified API
    'UnifiedDriftDetector',
    'DetectionMethod',
    'DriftType',
    'DriftResult',
    'DriftMonitor',
    'AlertRule',
    'AlertSeverity',
    'DriftAlert',
    
    # Legacy adapters
    'LegacyDataDriftDetector',
    'LegacyDriftDetector',
    
    # Old names for backward compatibility
    'DriftDetector',
    'DriftMetric',
    'DataDriftDetector'
]

# Set up logging
import logging
from typing import Optional

def configure_logging(level: int = logging.INFO, 
                     log_file: Optional[str] = None) -> None:
    """Configure logging for the monitoring module.
    
    Args:
        level: Logging level (default: logging.INFO)
        log_file: Optional file to write logs to (if None, logs to stderr)
    """
    logger = logging.getLogger('tick_analysis.monitoring')
    logger.setLevel(level)
    
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Add console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # Add file handler if specified
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    logger.info("Logging configured for tick_analysis.monitoring")

# Configure default logging when module is imported
configure_logging()
