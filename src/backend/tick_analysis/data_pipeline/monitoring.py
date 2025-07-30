"""
Advanced Monitoring Module

This module provides comprehensive monitoring capabilities including metrics
collection, alerting, health checks, and performance tracking.
"""

import logging
import asyncio
import time
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import os
import psutil
import aiohttp
from prometheus_client import Counter, Gauge, Histogram, start_http_server

logger = logging.getLogger(__name__)

@dataclass
class Alert:
    """Represents a monitoring alert."""
    name: str
    severity: str  # 'info', 'warning', 'error', 'critical'
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None

@dataclass
class HealthCheck:
    """Represents a health check result."""
    name: str
    status: str  # 'healthy', 'degraded', 'unhealthy'
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

class PipelineMonitor:
    """Advanced monitoring system for the data pipeline."""
    
    def __init__(self, 
                 pipeline_id: str,
                 metrics_port: int = 9090,
                 alert_handlers: Optional[List[Callable]] = None,
                 storage_path: str = './data/monitoring'):
        """
        Initialize the monitoring system.
        
        Args:
            pipeline_id: Unique identifier for the pipeline
            metrics_port: Port for Prometheus metrics
            alert_handlers: List of alert handler functions
            storage_path: Path to store monitoring data
        """
        self.pipeline_id = pipeline_id
        self.metrics_port = metrics_port
        self.alert_handlers = alert_handlers or []
        self.storage_path = storage_path
        
        # Create storage directory
        os.makedirs(storage_path, exist_ok=True)
        
        # Initialize Prometheus metrics
        self._init_metrics()
        
        # Alert state
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
        # Health check state
        self.health_checks: Dict[str, HealthCheck] = {}
        
        # Monitoring state
        self._running = False
        self._monitoring_task = None
        self._start_time = None
    
    def _init_metrics(self) -> None:
        """Initialize Prometheus metrics."""
        # Pipeline metrics
        self.metrics = {
            'processed_records': Counter(
                'pipeline_processed_records_total',
                'Total number of processed records',
                ['pipeline_id']
            ),
            'processing_errors': Counter(
                'pipeline_processing_errors_total',
                'Total number of processing errors',
                ['pipeline_id', 'error_type']
            ),
            'processing_latency': Histogram(
                'pipeline_processing_latency_seconds',
                'Record processing latency',
                ['pipeline_id']
            ),
            'batch_size': Gauge(
                'pipeline_batch_size',
                'Current batch size',
                ['pipeline_id']
            ),
            'queue_size': Gauge(
                'pipeline_queue_size',
                'Current queue size',
                ['pipeline_id']
            ),
            'memory_usage': Gauge(
                'pipeline_memory_usage_bytes',
                'Memory usage in bytes',
                ['pipeline_id']
            ),
            'cpu_usage': Gauge(
                'pipeline_cpu_usage_percent',
                'CPU usage percentage',
                ['pipeline_id']
            )
        }
    
    async def start(self) -> None:
        """Start the monitoring system."""
        if self._running:
            return
            
        # Start Prometheus metrics server
        start_http_server(self.metrics_port)
        
        self._running = True
        self._start_time = time.time()
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info(f"Pipeline monitoring started for pipeline {self.pipeline_id}")
    
    async def stop(self) -> None:
        """Stop the monitoring system."""
        if not self._running:
            return
            
        self._running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info(f"Pipeline monitoring stopped for pipeline {self.pipeline_id}")
    
    async def record_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Record pipeline metrics.
        
        Args:
            metrics: Dictionary of metrics to record
        """
        try:
            # Update Prometheus metrics
            if 'processed_records' in metrics:
                self.metrics['processed_records'].labels(
                    pipeline_id=self.pipeline_id
                ).inc(metrics['processed_records'])
            
            if 'processing_errors' in metrics:
                for error_type, count in metrics['processing_errors'].items():
                    self.metrics['processing_errors'].labels(
                        pipeline_id=self.pipeline_id,
                        error_type=error_type
                    ).inc(count)
            
            if 'processing_latency' in metrics:
                self.metrics['processing_latency'].labels(
                    pipeline_id=self.pipeline_id
                ).observe(metrics['processing_latency'])
            
            if 'batch_size' in metrics:
                self.metrics['batch_size'].labels(
                    pipeline_id=self.pipeline_id
                ).set(metrics['batch_size'])
            
            if 'queue_size' in metrics:
                self.metrics['queue_size'].labels(
                    pipeline_id=self.pipeline_id
                ).set(metrics['queue_size'])
            
            # Record system metrics
            process = psutil.Process()
            self.metrics['memory_usage'].labels(
                pipeline_id=self.pipeline_id
            ).set(process.memory_info().rss)
            
            self.metrics['cpu_usage'].labels(
                pipeline_id=self.pipeline_id
            ).set(process.cpu_percent())
            
        except Exception as e:
            logger.error(f"Error recording metrics: {e}")
    
    async def check_health(self) -> Dict[str, HealthCheck]:
        """
        Perform health checks.
        
        Returns:
            Dictionary of health check results
        """
        checks = {}
        
        try:
            # Check memory usage
            process = psutil.Process()
            memory_percent = process.memory_percent()
            checks['memory'] = HealthCheck(
                name='memory_usage',
                status='healthy' if memory_percent < 80 else 'degraded' if memory_percent < 90 else 'unhealthy',
                message=f"Memory usage: {memory_percent:.1f}%",
                metadata={'usage_percent': memory_percent}
            )
            
            # Check CPU usage
            cpu_percent = process.cpu_percent()
            checks['cpu'] = HealthCheck(
                name='cpu_usage',
                status='healthy' if cpu_percent < 70 else 'degraded' if cpu_percent < 85 else 'unhealthy',
                message=f"CPU usage: {cpu_percent:.1f}%",
                metadata={'usage_percent': cpu_percent}
            )
            
            # Check disk space
            disk_usage = psutil.disk_usage('/')
            checks['disk'] = HealthCheck(
                name='disk_usage',
                status='healthy' if disk_usage.percent < 80 else 'degraded' if disk_usage.percent < 90 else 'unhealthy',
                message=f"Disk usage: {disk_usage.percent:.1f}%",
                metadata={'usage_percent': disk_usage.percent}
            )
            
            # Update health check state
            self.health_checks.update(checks)
            
            # Persist health checks
            await self._persist_health_checks()
            
        except Exception as e:
            logger.error(f"Error performing health checks: {e}")
        
        return checks
    
    async def create_alert(self, 
                          name: str,
                          severity: str,
                          message: str,
                          metadata: Optional[Dict[str, Any]] = None) -> Alert:
        """
        Create and handle a new alert.
        
        Args:
            name: Alert name
            severity: Alert severity
            message: Alert message
            metadata: Additional alert metadata
            
        Returns:
            Created alert
        """
        alert = Alert(
            name=name,
            severity=severity,
            message=message,
            metadata=metadata or {}
        )
        
        # Store alert
        self.active_alerts[name] = alert
        self.alert_history.append(alert)
        
        # Persist alerts
        await self._persist_alerts()
        
        # Handle alert
        for handler in self.alert_handlers:
            try:
                await handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
        
        return alert
    
    async def resolve_alert(self, name: str) -> Optional[Alert]:
        """
        Resolve an active alert.
        
        Args:
            name: Name of the alert to resolve
            
        Returns:
            Resolved alert or None if not found
        """
        if name not in self.active_alerts:
            return None
        
        alert = self.active_alerts[name]
        alert.resolved = True
        alert.resolved_at = datetime.utcnow()
        
        # Remove from active alerts
        del self.active_alerts[name]
        
        # Persist alerts
        await self._persist_alerts()
        
        return alert
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                # Perform health checks
                health_checks = await self.check_health()
                
                # Check for unhealthy conditions
                for check in health_checks.values():
                    if check.status == 'unhealthy':
                        await self.create_alert(
                            name=f"health_check_{check.name}",
                            severity='critical',
                            message=f"Health check failed: {check.message}",
                            metadata=check.metadata
                        )
                    elif check.status == 'degraded':
                        await self.create_alert(
                            name=f"health_check_{check.name}",
                            severity='warning',
                            message=f"Health check degraded: {check.message}",
                            metadata=check.metadata
                        )
                
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(1)
    
    async def _persist_alerts(self) -> None:
        """Persist alerts to disk."""
        try:
            # Save active alerts
            active_alerts_file = os.path.join(self.storage_path, 'active_alerts.json')
            async with aiofiles.open(active_alerts_file, 'w') as f:
                await f.write(json.dumps({
                    name: alert.to_dict()
                    for name, alert in self.active_alerts.items()
                }, indent=2))
            
            # Save alert history
            history_file = os.path.join(self.storage_path, 'alert_history.json')
            async with aiofiles.open(history_file, 'w') as f:
                await f.write(json.dumps([
                    alert.to_dict()
                    for alert in self.alert_history[-1000:]  # Keep last 1000 alerts
                ], indent=2))
                
        except Exception as e:
            logger.error(f"Error persisting alerts: {e}")
    
    async def _persist_health_checks(self) -> None:
        """Persist health checks to disk."""
        try:
            health_checks_file = os.path.join(self.storage_path, 'health_checks.json')
            async with aiofiles.open(health_checks_file, 'w') as f:
                await f.write(json.dumps({
                    name: {
                        'status': check.status,
                        'message': check.message,
                        'timestamp': check.timestamp.isoformat(),
                        'metadata': check.metadata
                    }
                    for name, check in self.health_checks.items()
                }, indent=2))
                
        except Exception as e:
            logger.error(f"Error persisting health checks: {e}")
    
    def get_monitoring_info(self) -> Dict[str, Any]:
        """Get current monitoring information."""
        return {
            'pipeline_id': self.pipeline_id,
            'uptime': time.time() - self._start_time if self._start_time else 0,
            'active_alerts': len(self.active_alerts),
            'health_checks': {
                name: {
                    'status': check.status,
                    'message': check.message,
                    'timestamp': check.timestamp.isoformat()
                }
                for name, check in self.health_checks.items()
            }
        } 