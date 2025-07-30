"""
Advanced metrics collection system for Tick Data Analysis & Alpha Detection.
"""

from typing import Dict, List, Optional
import time
from datetime import datetime
import pandas as pd
import numpy as np
from prometheus_client import Counter, Gauge, Histogram, Summary
import logging

logger = logging.getLogger(__name__)

class SystemMetrics:
    """Advanced metrics collection system."""
    
    def __init__(self):
        """Initialize metrics collectors."""
        # System metrics
        self.cpu_usage = Gauge('system_cpu_usage', 'CPU usage percentage')
        self.memory_usage = Gauge('system_memory_usage', 'Memory usage percentage')
        self.disk_usage = Gauge('system_disk_usage', 'Disk usage percentage')
        
        # Application metrics
        self.request_count = Counter('app_request_count', 'Total request count')
        self.request_latency = Histogram('app_request_latency', 'Request latency')
        self.error_count = Counter('app_error_count', 'Total error count')
        
        # Business metrics
        self.trade_count = Counter('business_trade_count', 'Total trade count')
        self.trade_volume = Counter('business_trade_volume', 'Total trade volume')
        self.profit_loss = Gauge('business_profit_loss', 'Current P&L')
        
        # Performance metrics
        self.data_processing_time = Summary('perf_data_processing_time', 
                                          'Data processing time')
        self.strategy_execution_time = Summary('perf_strategy_execution_time',
                                             'Strategy execution time')
        
    def collect_system_metrics(self) -> Dict[str, float]:
        """Collect system metrics.
        
        Returns:
            Dictionary of system metrics
        """
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent()
            self.cpu_usage.set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.memory_usage.set(memory.percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.disk_usage.set(disk.percent)
            
            return {
                'cpu_usage': cpu_percent,
                'memory_usage': memory.percent,
                'disk_usage': disk.percent
            }
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return {}
            
    def record_request(self, latency: float) -> None:
        """Record request metrics.
        
        Args:
            latency: Request latency in seconds
        """
        self.request_count.inc()
        self.request_latency.observe(latency)
        
    def record_error(self) -> None:
        """Record error metrics."""
        self.error_count.inc()
        
    def record_trade(self, volume: float, pnl: float) -> None:
        """Record trade metrics.
        
        Args:
            volume: Trade volume
            pnl: Profit/loss
        """
        self.trade_count.inc()
        self.trade_volume.inc(volume)
        self.profit_loss.set(pnl)
        
    def record_data_processing_time(self, processing_time: float) -> None:
        """Record data processing time.
        
        Args:
            processing_time: Processing time in seconds
        """
        self.data_processing_time.observe(processing_time)
        
    def record_strategy_execution_time(self, execution_time: float) -> None:
        """Record strategy execution time.
        
        Args:
            execution_time: Execution time in seconds
        """
        self.strategy_execution_time.observe(execution_time)
        
    def get_metrics_summary(self) -> Dict[str, Dict[str, float]]:
        """Get metrics summary.
        
        Returns:
            Dictionary of metrics summaries
        """
        return {
            'system': self.collect_system_metrics(),
            'application': {
                'request_count': self.request_count._value.get(),
                'error_count': self.error_count._value.get(),
                'avg_latency': self.request_latency._sum.get() / 
                              max(self.request_count._value.get(), 1)
            },
            'business': {
                'trade_count': self.trade_count._value.get(),
                'trade_volume': self.trade_volume._value.get(),
                'profit_loss': self.profit_loss._value.get()
            },
            'performance': {
                'avg_data_processing_time': 
                    self.data_processing_time._sum.get() / 
                    max(self.data_processing_time._count.get(), 1),
                'avg_strategy_execution_time':
                    self.strategy_execution_time._sum.get() /
                    max(self.strategy_execution_time._count.get(), 1)
            }
        } 