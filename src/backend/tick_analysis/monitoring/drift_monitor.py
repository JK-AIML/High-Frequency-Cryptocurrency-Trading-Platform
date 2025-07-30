"""
Real-time Drift Monitoring and Alerting

This module provides a monitoring system for detecting data drift in real-time
streaming data and triggering alerts when drift is detected.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Callable, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from collections import deque, defaultdict

from .unified_drift import (
    UnifiedDriftDetector,
    DriftResult,
    DriftType,
    DetectionMethod
)

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Severity levels for drift alerts."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"

@dataclass
class DriftAlert:
    """Container for drift alert information."""
    timestamp: datetime
    severity: AlertSeverity
    message: str
    drift_result: DriftResult
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'severity': self.severity.value,
            'message': self.message,
            'drift_result': self.drift_result.to_dict(),
            'metadata': self.metadata
        }
    
    def to_json(self) -> str:
        """Convert alert to JSON string."""
        return json.dumps(self.to_dict(), default=str)

class AlertRule:
    """Rule for triggering alerts based on drift detection results."""
    
    def __init__(
        self,
        name: str,
        condition: Callable[[DriftResult], bool],
        severity: AlertSeverity = AlertSeverity.WARNING,
        cooldown: int = 300,  # seconds
        message_template: Optional[str] = None
    ):
        """
        Initialize an alert rule.
        
        Args:
            name: Name of the alert rule
            condition: Function that takes a DriftResult and returns True if alert should trigger
            severity: Severity level of the alert
            cooldown: Minimum time between alerts in seconds
            message_template: Template for alert message (can use {field} placeholders)
        """
        self.name = name
        self.condition = condition
        self.severity = severity
        self.cooldown = cooldown
        self.message_template = message_template or "Drift detected: {feature} - {message}"
        self.last_triggered: Optional[datetime] = None
    
    def should_trigger(self, result: DriftResult) -> bool:
        """Check if the rule should trigger for the given drift result."""
        if self.condition(result):
            now = datetime.utcnow()
            if self.last_triggered is None or (now - self.last_triggered).total_seconds() > self.cooldown:
                self.last_triggered = now
                return True
        return False
    
    def create_alert(self, result: DriftResult, metadata: Optional[Dict] = None) -> DriftAlert:
        """Create an alert from a drift result."""
        message = self.message_template.format(
            feature=result.feature or 'unknown',
            message=result.message,
            statistic=result.statistic,
            p_value=result.p_value,
            threshold=result.threshold
        )
        return DriftAlert(
            timestamp=datetime.utcnow(),
            severity=self.severity,
            message=message,
            drift_result=result,
            metadata=metadata or {}
        )

class DriftMonitor:
    """
    Real-time drift monitoring system.
    
    This class provides a framework for monitoring data drift in real-time
    and triggering alerts when drift is detected.
    """
    
    def __init__(
        self,
        reference_data: Union[pd.DataFrame, np.ndarray],
        window_size: int = 1000,
        min_samples: int = 100,
        alpha: float = 0.05,
        alert_rules: Optional[List[AlertRule]] = None,
        **detector_kwargs
    ):
        """
        Initialize the drift monitor.
        
        Args:
            reference_data: Reference dataset (baseline)
            window_size: Size of the sliding window for drift detection
            min_samples: Minimum samples required before performing drift detection
            alpha: Significance level for drift detection
            alert_rules: List of alert rules to use
            **detector_kwargs: Additional arguments for UnifiedDriftDetector
        """
        self.detector = UnifiedDriftDetector(reference_data, **detector_kwargs)
        self.window_size = window_size
        self.min_samples = min_samples
        self.alpha = alpha
        self.data_buffer = defaultdict(lambda: deque(maxlen=window_size))
        self.alert_rules = alert_rules or self._get_default_alert_rules()
        self.alert_handlers = []
    
    def _get_default_alert_rules(self) -> List[AlertRule]:
        """Get default alert rules."""
        return [
            # Alert on significant drift with high confidence
            AlertRule(
                name="high_confidence_drift",
                condition=lambda r: r.p_value is not None and r.p_value < 0.01,
                severity=AlertSeverity.CRITICAL,
                message_template="Critical drift detected in {feature}: p={p_value:.4f} (threshold=0.01)"
            ),
            # Alert on moderate drift
            AlertRule(
                name="moderate_drift",
                condition=lambda r: r.p_value is not None and r.p_value < 0.05,
                severity=AlertSeverity.WARNING,
                message_template="Warning: Drift detected in {feature}: p={p_value:.4f}",
                cooldown=600  # 10 minutes
            ),
            # Alert on large effect size
            AlertRule(
                name="large_effect_size",
                condition=lambda r: r.statistic > 0.5,  # Adjust threshold as needed
                severity=AlertSeverity.WARNING,
                message_template="Large effect size detected in {feature}: {statistic:.2f}"
            )
        ]
    
    def add_alert_handler(self, handler: Callable[[DriftAlert], None]) -> None:
        """Add an alert handler function."""
        self.alert_handlers.append(handler)
    
    async def _handle_alert(self, alert: DriftAlert) -> None:
        """Handle an alert by calling all registered handlers."""
        for handler in self.alert_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}", exc_info=True)
    
    def update(self, data: Dict[str, Any], metadata: Optional[Dict] = None) -> List[DriftAlert]:
        """
        Update the monitor with new data and check for drift.
        
        Args:
            data: Dictionary of feature-value pairs
            metadata: Optional metadata to include in alerts
            
        Returns:
            List of triggered alerts
        """
        alerts = []
        
        # Update data buffers
        for feature, value in data.items():
            self.data_buffer[feature].append(value)
        
        # Check if we have enough samples
        if len(self.data_buffer) == 0 or len(next(iter(self.data_buffer.values()))) < self.min_samples:
            return []
        
        # Convert buffer to DataFrame for detection
        current_data = pd.DataFrame({
            feature: list(values) 
            for feature, values in self.data_buffer.items()
        })
        
        # Detect drift
        results = self.detector.detect_drift(
            current_data,
            alpha=self.alpha,
            method=DetectionMethod.KOLMOGOROV_SMIRNOV
        )
        
        # Check for alerts
        for result in results:
            if not result.is_drifted:
                continue
                
            for rule in self.alert_rules:
                if rule.should_trigger(result):
                    alert = rule.create_alert(result, metadata)
                    alerts.append(alert)
                    # Handle alert asynchronously
                    asyncio.create_task(self._handle_alert(alert))
        
        return alerts
    
    async def monitor_stream(
        self,
        data_stream: 'AsyncIterator[Dict[str, Any]]',
        batch_size: int = 100,
        batch_timeout: float = 1.0
    ) -> None:
        """
        Monitor a stream of data for drift.
        
        Args:
            data_stream: Async iterator yielding data points
            batch_size: Number of points to collect before checking for drift
            batch_timeout: Maximum time to wait for a batch in seconds
        """
        batch = []
        last_batch_time = datetime.utcnow()
        
        try:
            async for data_point in data_stream:
                batch.append(data_point)
                
                # Check if we should process the batch
                time_since_last_batch = (datetime.utcnow() - last_batch_time).total_seconds()
                if len(batch) >= batch_size or time_since_last_batch >= batch_timeout:
                    if batch:
                        # Process batch
                        for point in batch:
                            await self.update(point)
                        batch = []
                        last_batch_time = datetime.utcnow()
        except Exception as e:
            logger.error(f"Error in monitor_stream: {e}", exc_info=True)
            raise

# Example usage
if __name__ == "__main__":
    import numpy as np
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Generate some example reference data
    np.random.seed(42)
    ref_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.normal(10, 2, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    })
    
    # Create monitor with default alert rules
    monitor = DriftMonitor(
        reference_data=ref_data,
        categorical_features=['category'],
        window_size=500,
        min_samples=100
    )
    
    # Add a simple alert handler
    def log_alert(alert: DriftAlert):
        print(f"\n[ALERT {alert.severity.value}] {alert.timestamp}")
        print(f"Message: {alert.message}")
        print(f"Feature: {alert.drift_result.feature}")
        print(f"Statistic: {alert.drift_result.statistic:.4f}")
        print(f"P-value: {alert.drift_result.p_value:.4f}")
    
    monitor.add_alert_handler(log_alert)
    
    # Simulate a data stream with drift
    print("Starting drift monitoring...")
    print("Press Ctrl+C to stop")
    
    try:
        i = 0
        while True:
            # Simulate data with gradual drift in feature1
            data = {
                'feature1': np.random.normal(0.1 * (i // 100), 1.0),
                'feature2': np.random.normal(10, 2),
                'category': np.random.choice(['A', 'B', 'C'])
            }
            
            # Update monitor
            monitor.update(data)
            
            # Print progress
            if i % 100 == 0:
                print(f"Processed {i} data points...")
            
            i += 1
            
            # Add some delay to simulate real-time data
            import time
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nStopping drift monitor...")
