"""
Monitoring and Alerting System

Provides real-time monitoring, metrics collection, and alerting for the trading system.
"""

import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from collections import defaultdict, deque
import threading
import numpy as np
import pandas as pd
from prometheus_client import (
    start_http_server,
    Counter,
    Gauge,
    Histogram,
    Summary,
    CollectorRegistry
)

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Severity levels for alerts."""
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()

class AlertType(Enum):
    """Types of alerts that can be generated."""
    DATA_QUALITY = auto()
    DRIFT_DETECTED = auto()
    PERFORMANCE = auto()
    SYSTEM = auto()
    BUSINESS = auto()

@dataclass
class Alert:
    """Represents an alert condition."""
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary for serialization."""
        return {
            'alert_type': self.alert_type.name,
            'severity': self.severity.name,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }

class AlertHandler(Protocol):
    """Protocol for alert handlers."""
    def handle_alert(self, alert: Alert) -> None: ...

class MetricType(Enum):
    """Types of metrics that can be tracked."""
    COUNTER = auto()
    GAUGE = auto()
    HISTOGRAM = auto()
    SUMMARY = auto()

@dataclass
class MetricDefinition:
    """Definition of a metric to be tracked."""
    name: str
    metric_type: MetricType
    description: str = ""
    labels: Optional[List[str]] = None
    buckets: Optional[List[float]] = None
    unit: str = ""

class MetricsCollector:
    """Collects and manages system and application metrics."""
    
    def __init__(self, port: int = 8000, enable_http: bool = True):
        """Initialize the metrics collector."""
        self.registry = CollectorRegistry()
        self.metrics: Dict[str, Any] = {}
        self.alert_handlers: List[AlertHandler] = []
        self.port = port
        self._started = False
        
        if enable_http:
            self._start_http_server()
    
    def _start_http_server(self) -> None:
        """Start the Prometheus HTTP server."""
        if not self._started:
            start_http_server(self.port, registry=self.registry)
            self._started = True
            logger.info(f"Metrics server started on port {self.port}")
    
    def register_metric(self, metric_def: MetricDefinition) -> None:
        """Register a new metric."""
        if metric_def.name in self.metrics:
            return
        
        labels = metric_def.labels or []
        
        if metric_def.metric_type == MetricType.COUNTER:
            self.metrics[metric_def.name] = Counter(
                metric_def.name,
                metric_def.description,
                labelnames=labels,
                registry=self.registry
            )
        elif metric_def.metric_type == MetricType.GAUGE:
            self.metrics[metric_def.name] = Gauge(
                metric_def.name,
                metric_def.description,
                labelnames=labels,
                registry=self.registry
            )
        elif metric_def.metric_type == MetricType.HISTOGRAM:
            buckets = metric_def.buckets or (
                0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0
            )
            self.metrics[metric_def.name] = Histogram(
                metric_def.name,
                metric_def.description,
                labelnames=labels,
                buckets=buckets,
                registry=self.registry
            )
        elif metric_def.metric_type == MetricType.SUMMARY:
            self.metrics[metric_def.name] = Summary(
                f"{metric_def.name}_{metric_def.unit}",
                metric_def.description,
                labelnames=labels,
                registry=self.registry
            )
    
    def record_metric(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> None:
        """Record a metric value."""
        if name not in self.metrics:
            logger.warning(f"Metric {name} not registered")
            return
        
        metric = self.metrics[name]
        label_values = labels or {}
        
        if hasattr(metric, 'labels'):
            metric = metric.labels(**label_values)
        
        if hasattr(metric, 'observe'):
            metric.observe(value)
        elif hasattr(metric, 'inc'):
            metric.inc(value)
        elif hasattr(metric, 'set'):
            metric.set(value)
        elif hasattr(metric, 'dec'):
            metric.dec(value)
    
    def add_alert_handler(self, handler: AlertHandler) -> None:
        """Add an alert handler."""
        self.alert_handlers.append(handler)
    
    def trigger_alert(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        message: str,
        **metadata
    ) -> None:
        """Trigger an alert."""
        alert = Alert(
            alert_type=alert_type,
            severity=severity,
            message=message,
            metadata=metadata
        )
        
        for handler in self.alert_handlers:
            try:
                handler.handle_alert(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}", exc_info=True)

class ConsoleAlertHandler:
    """Simple alert handler that logs to console."""
    
    def handle_alert(self, alert: Alert) -> None:
        """Handle an alert by logging it to the console."""
        log_level = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.ERROR: logging.ERROR,
            AlertSeverity.CRITICAL: logging.CRITICAL
        }.get(alert.severity, logging.INFO)
        
        logger.log(
            log_level,
            f"[{alert.alert_type.name}] {alert.severity.name}: {alert.message}",
            extra=alert.metadata
        )

class SlackAlertHandler:
    """Alert handler that sends alerts to Slack."""
    
    def __init__(self, webhook_url: str, channel: str = "#alerts"):
        """Initialize with Slack webhook URL and channel."""
        self.webhook_url = webhook_url
        self.channel = channel
    
    def handle_alert(self, alert: Alert) -> None:
        """Send alert to Slack."""
        try:
            import requests
            
            color = {
                AlertSeverity.INFO: "#36a64f",
                AlertSeverity.WARNING: "#f2c744",
                AlertSeverity.ERROR: "#ff6b6b",
                AlertSeverity.CRITICAL: "#ff0000"
            }.get(alert.severity, "#439FE0")
            
            payload = {
                "channel": self.channel,
                "attachments": [{
                    "color": color,
                    "title": f"{alert.alert_type.name}: {alert.severity.name}",
                    "text": alert.message,
                    "fields": [
                        {
                            "title": "Timestamp",
                            "value": alert.timestamp.isoformat(),
                            "short": True
                        },
                        {
                            "title": "Severity",
                            "value": alert.severity.name,
                            "short": True
                        }
                    ],
                    "mrkdwn_in": ["text", "fields"]
                }]
            }
            
            # Add metadata as fields
            for key, value in alert.metadata.items():
                payload["attachments"][0]["fields"].append({
                    "title": key,
                    "value": str(value)[:100],
                    "short": True
                })
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}", exc_info=True)

class MetricsAggregator:
    """Aggregates metrics over time windows."""
    
    def __init__(
        self,
        window_size: int = 60,  # seconds
        max_samples: int = 1000
    ):
        """Initialize the aggregator with window size and max samples."""
        self.window_size = window_size
        self.max_samples = max_samples
        self.metrics: Dict[str, Dict[Tuple, deque]] = {}
        self.lock = threading.RLock()
    
    def add_metric(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Add a metric value with optional labels."""
        if labels is None:
            labels = {}
        
        key = tuple(sorted(labels.items()))
        timestamp = time.time()
        
        with self.lock:
            if name not in self.metrics:
                self.metrics[name] = {}
            
            if key not in self.metrics[name]:
                self.metrics[name][key] = deque(maxlen=self.max_samples)
            
            self.metrics[name][key].append((timestamp, value))
    
    def get_aggregates(
        self,
        name: str,
        window: Optional[int] = None,
        labels: Optional[Dict[str, str]] = None
    ) -> Dict[str, float]:
        """Get aggregated metrics for a given name and optional labels."""
        window = window or self.window_size
        now = time.time()
        
        with self.lock:
            if name not in self.metrics:
                return {}
            
            result = {}
            
            for key, values in self.metrics[name].items():
                # Filter by time window
                filtered = [
                    v for t, v in values
                    if t >= now - window
                ]
                
                if not filtered:
                    continue
                
                # Apply label filtering if specified
                label_dict = dict(key)
                if labels and not all(
                    label_dict.get(k) == v for k, v in labels.items()
                ):
                    continue
                
                # Calculate aggregates
                arr = np.array(filtered)
                result_key = ",".join(f"{k}={v}" for k, v in label_dict.items())
                
                result[f"{name}_count_{result_key}"] = len(arr)
                result[f"{name}_sum_{result_key}"] = float(np.sum(arr))
                result[f"{name}_avg_{result_key}"] = float(np.mean(arr))
                result[f"{name}_min_{result_key}"] = float(np.min(arr))
                result[f"{name}_max_{result_key}"] = float(np.max(arr))
                result[f"{name}_p50_{result_key}"] = float(np.percentile(arr, 50))
                result[f"{name}_p95_{result_key}"] = float(np.percentile(arr, 95))
                result[f"{name}_p99_{result_key}"] = float(np.percentile(arr, 99))
                
                if len(arr) > 1:
                    result[f"{name}_std_{result_key}"] = float(np.std(arr))
                else:
                    result[f"{name}_std_{result_key}"] = 0.0
            
            return result
    
    def clear_old_entries(self, max_age: int = 3600) -> None:
        """Clear entries older than max_age seconds."""
        now = time.time()
        
        with self.lock:
            for name in list(self.metrics.keys()):
                for key in list(self.metrics[name].keys()):
                    # Remove old entries from each deque
                    q = self.metrics[name][key]
                    while q and q[0][0] < now - max_age:
                        q.popleft()
                
                # Remove empty label sets
                self.metrics[name] = {
                    k: v for k, v in self.metrics[name].items()
                    if v  # Only keep non-empty deques
                }
            
            # Remove empty metric names
            self.metrics = {
                k: v for k, v in self.metrics.items()
                if v  # Only keep non-empty label sets
            }

# Default metrics collector instance
metrics = MetricsCollector(enable_http=True)

# Register common metrics
metrics.register_metric(
    MetricDefinition(
        name="tick_processing_duration_seconds",
        metric_type=MetricType.HISTOGRAM,
        description="Time taken to process a tick",
        unit="seconds"
    )
)

metrics.register_metric(
    MetricDefinition(
        name="tick_processing_errors_total",
        metric_type=MetricType.COUNTER,
        description="Total number of tick processing errors",
        labels=["error_type"]
    )
)

metrics.register_metric(
    MetricDefinition(
        name="data_quality_issues_total",
        metric_type=MetricType.COUNTER,
        description="Total number of data quality issues",
        labels=["check_type", "field"]
    )
)

metrics.register_metric(
    MetricDefinition(
        name="drift_detection_alerts_total",
        metric_type=MetricType.COUNTER,
        description="Total number of drift detection alerts",
        labels=["drift_type", "severity"]
    )
)
