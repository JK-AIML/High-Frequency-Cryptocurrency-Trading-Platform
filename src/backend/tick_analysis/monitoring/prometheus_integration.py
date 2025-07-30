"""
Prometheus and Grafana integration for monitoring drift detection.

This module provides functionality to expose metrics to Prometheus and Grafana
for monitoring the drift detection system.
"""

import time
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

from prometheus_client import (
    start_http_server,
    Counter,
    Gauge,
    Histogram,
    Summary,
    CollectorRegistry,
    push_to_gateway,
    pushadd_to_gateway,
    delete_from_gateway,
)
from prometheus_client.exposition import basic_auth_handler

from ..streaming.operators import StreamOperator
from ..exceptions import StreamProcessingError

logger = logging.getLogger(__name__)

@dataclass
class PrometheusMetricsConfig:
    """Configuration for Prometheus metrics."""
    
    # General settings
    namespace: str = "tick_analysis"
    subsystem: str = "drift_detection"
    
    # Push gateway settings (if using push gateway)
    push_gateway_url: Optional[str] = None
    job_name: str = "drift_monitor"
    username: Optional[str] = None
    password: Optional[str] = None
    
    # HTTP server settings (if using pull model)
    http_port: int = 8000
    
    # Metrics collection interval in seconds
    collection_interval: int = 15
    
    # Labels to add to all metrics
    static_labels: Dict[str, str] = field(default_factory=dict)


class PrometheusMetricsExporter:
    """Exports metrics to Prometheus."""
    
    def __init__(self, config: PrometheusMetricsConfig):
        """Initialize the metrics exporter."""
        self.config = config
        self.registry = CollectorRegistry()
        self._metrics: Dict[str, Any] = {}
        self._running = False
        self._executor = ThreadPoolExecutor(max_workers=1)
        
        # Common labels for all metrics
        self.common_labels = {
            **config.static_labels,
            "job": config.job_name,
        }
        
        # Initialize metrics
        self._init_metrics()
    
    def _init_metrics(self):
        """Initialize Prometheus metrics."""
        # Drift detection metrics
        self._metrics['drift_detections_total'] = Counter(
            f'{self.config.namespace}_{self.config.subsystem}_drift_detections_total',
            'Total number of drift detections',
            ['feature', 'drift_type'],
            registry=self.registry,
            **self.common_labels
        )
        
        self._metrics['drift_score'] = Gauge(
            f'{self.config.namespace}_{self.config.subsystem}_drift_score',
            'Current drift score',
            ['feature'],
            registry=self.registry,
            **self.common_labels
        )
        
        self._metrics['p_value'] = Gauge(
            f'{self.config.namespace}_{self.config.subsystem}_p_value',
            'P-value from statistical tests',
            ['feature', 'test_type'],
            registry=self.registry,
            **self.common_labels
        )
        
        # Performance metrics
        self._metrics['processing_duration_seconds'] = Histogram(
            f'{self.config.namespace}_{self.config.subsystem}_processing_duration_seconds',
            'Time spent processing data',
            ['pipeline_stage'],
            registry=self.registry,
            **self.common_labels,
            buckets=(.001, .0025, .005, .01, .025, .05, .075, .1, .25, .5, 1.0, 2.5, 5.0, 10.0)
        )
        
        # Alert metrics
        self._metrics['alerts_total'] = Counter(
            f'{self.config.namespace}_{self.config.subsystem}_alerts_total',
            'Total number of alerts',
            ['severity', 'feature'],
            registry=self.registry,
            **self.common_labels
        )
        
        # System metrics
        self._metrics['system_cpu_usage'] = Gauge(
            f'{self.config.namespace}_{self.config.subsystem}_system_cpu_usage',
            'System CPU usage percentage',
            registry=self.registry,
            **self.common_labels
        )
        
        self._metrics['system_memory_usage'] = Gauge(
            f'{self.config.namespace}_{self.config.subsystem}_system_memory_usage',
            'System memory usage percentage',
            registry=self.registry,
            **self.common_labels
        )
        
        # Custom metrics dictionary
        self._custom_metrics: Dict[str, Any] = {}
    
    def record_drift_detection(
        self,
        feature: str,
        drift_type: str,
        score: float,
        p_value: float,
        test_type: str = "ks_test"
    ) -> None:
        """Record drift detection results."""
        self._metrics['drift_detections_total'].labels(
            feature=feature,
            drift_type=drift_type
        ).inc()
        
        self._metrics['drift_score'].labels(feature=feature).set(score)
        
        self._metrics['p_value'].labels(
            feature=feature,
            test_type=test_type
        ).set(p_value)
    
    def record_processing_time(self, stage: str, duration: float) -> None:
        """Record processing time for a pipeline stage."""
        self._metrics['processing_duration_seconds'].labels(
            pipeline_stage=stage
        ).observe(duration)
    
    def record_alert(
        self,
        severity: str,
        feature: str,
        alert_type: str = "drift"
    ) -> None:
        """Record an alert."""
        self._metrics['alerts_total'].labels(
            severity=severity,
            feature=feature
        ).inc()
    
    def update_system_metrics(self) -> Dict[str, float]:
        """Update system metrics."""
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent()
            self._metrics['system_cpu_usage'].set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self._metrics['system_memory_usage'].set(memory.percent)
            
            return {
                'cpu_usage': cpu_percent,
                'memory_usage': memory.percent,
            }
            
        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")
            return {}
    
    def create_custom_metric(
        self,
        name: str,
        metric_type: str,
        description: str = "",
        labels: Optional[List[str]] = None,
        **kwargs
    ) -> Any:
        """Create a custom Prometheus metric.
        
        Args:
            name: Metric name (will be prefixed with namespace and subsystem)
            metric_type: One of 'counter', 'gauge', 'histogram', 'summary'
            description: Metric description
            labels: List of label names
            **kwargs: Additional arguments to pass to the metric constructor
            
        Returns:
            The created metric
        """
        if labels is None:
            labels = []
            
        full_name = f"{self.config.namespace}_{self.config.subsystem}_{name}"
        
        if metric_type == 'counter':
            metric = Counter(
                full_name,
                description,
                labels,
                registry=self.registry,
                **self.common_labels,
                **kwargs
            )
        elif metric_type == 'gauge':
            metric = Gauge(
                full_name,
                description,
                labels,
                registry=self.registry,
                **self.common_labels,
                **kwargs
            )
        elif metric_type == 'histogram':
            metric = Histogram(
                full_name,
                description,
                labels,
                registry=self.registry,
                **self.common_labels,
                **kwargs
            )
        elif metric_type == 'summary':
            metric = Summary(
                full_name,
                description,
                labels,
                registry=self.registry,
                **self.common_labels,
                **kwargs
            )
        else:
            raise ValueError(f"Invalid metric type: {metric_type}")
        
        self._custom_metrics[name] = metric
        return metric
    
    def get_metric(self, name: str) -> Any:
        """Get a metric by name."""
        if name in self._metrics:
            return self._metrics[name]
        return self._custom_metrics.get(name)
    
    def start_http_server(self, port: Optional[int] = None) -> None:
        """Start an HTTP server to expose metrics."""
        port = port or self.config.http_port
        start_http_server(port, registry=self.registry)
        logger.info(f"Prometheus metrics server started on port {port}")
    
    async def push_metrics_async(self) -> None:
        """Push metrics to the PushGateway asynchronously."""
        if not self.config.push_gateway_url:
            logger.warning("Push gateway URL not configured")
            return
            
        def push():
            try:
                auth_handler = None
                if self.config.username and self.config.password:
                    auth_handler = basic_auth_handler(
                        '',
                        self.config.username,
                        self.config.password
                    )
                
                push_to_gateway(
                    self.config.push_gateway_url,
                    job=self.config.job_name,
                    registry=self.registry,
                    handler=auth_handler
                )
                logger.debug("Successfully pushed metrics to gateway")
            except Exception as e:
                logger.error(f"Error pushing metrics to gateway: {e}")
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self._executor, push)
    
    def start_periodic_push(self, interval: Optional[int] = None) -> None:
        """Start periodically pushing metrics to the PushGateway."""
        if not self.config.push_gateway_url:
            logger.warning("Push gateway URL not configured")
            return
            
        self._running = True
        interval = interval or self.config.collection_interval
        
        async def _push_loop():
            while self._running:
                try:
                    await self.push_metrics_async()
                    await asyncio.sleep(interval)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in metrics push loop: {e}")
                    await asyncio.sleep(min(60, interval))  # Back off on error
        
        self._push_task = asyncio.create_task(_push_loop())
        logger.info(f"Started periodic metrics push every {interval} seconds")
    
    def stop_periodic_push(self) -> None:
        """Stop the periodic push of metrics."""
        self._running = False
        if hasattr(self, '_push_task'):
            self._push_task.cancel()
            logger.info("Stopped periodic metrics push")
    
    def __del__(self):
        """Clean up resources."""
        self.stop_periodic_push()
        self._executor.shutdown(wait=True)


class DriftMetricsExporter(StreamOperator[Dict[str, Any]]):
    """Stream operator that exports drift metrics to Prometheus."""
    
    def __init__(self, exporter: PrometheusMetricsExporter):
        """Initialize the metrics exporter operator.
        
        Args:
            exporter: PrometheusMetricsExporter instance
        """
        self.exporter = exporter
        self._last_update = time.time()
    
    def process(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process a stream item and export metrics."""
        try:
            # Record processing time
            current_time = time.time()
            processing_time = current_time - self._last_update
            self._last_update = current_time
            
            # Record metrics for drift detection results
            if '_drift' in item and isinstance(item['_drift'], dict):
                drift_data = item['_drift']
                
                if 'results' in drift_data and isinstance(drift_data['results'], list):
                    for result in drift_data['results']:
                        self.exporter.record_drift_detection(
                            feature=result.get('feature', 'unknown'),
                            drift_type=result.get('drift_type', 'unknown'),
                            score=float(result.get('statistic', 0)),
                            p_value=float(result.get('p_value', 1.0)),
                            test_type=result.get('test_type', 'unknown')
                        )
            
            # Record alerts
            if '_alerts' in item and isinstance(item['_alerts'], list):
                for alert in item['_alerts']:
                    self.exporter.record_alert(
                        severity=alert.get('severity', 'info').lower(),
                        feature=alert.get('feature', 'unknown'),
                        alert_type=alert.get('type', 'drift')
                    )
            
            # Record processing time
            self.exporter.record_processing_time(
                stage='drift_detection',
                duration=processing_time
            )
            
            # Update system metrics periodically
            if current_time - getattr(self, '_last_system_update', 0) > 60:  # Every minute
                self.exporter.update_system_metrics()
                self._last_system_update = current_time
            
            return item
            
        except Exception as e:
            raise StreamProcessingError(f"Error in metrics export: {str(e)}") from e
