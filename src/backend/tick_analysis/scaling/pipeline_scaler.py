"""Dynamic scaling integration for streaming pipelines."""
from typing import Dict, Any, Optional, Callable
import time
import logging
from dataclasses import dataclass
from enum import Enum
from typing import List

from prometheus_client import Gauge, Counter

from ..streaming.operators import StreamPipeline
from .auto_scaler import AutoScaler, ScalingMetric

logger = logging.getLogger(__name__)

class PipelineScaler:
    """Manages dynamic scaling of a streaming pipeline."""
    
    def __init__(
        self,
        pipeline: StreamPipeline,
        min_workers: int = 1,
        max_workers: int = 100,
        metrics_interval: float = 5.0,
        scale_up_threshold: float = 0.7,
        scale_down_threshold: float = 0.3,
        cooldown: float = 60.0,
        metrics_provider: Optional[Callable[[], Dict[str, float]]] = None
    ):
        """Initialize the pipeline scaler.
        
        Args:
            pipeline: The stream pipeline to scale
            min_workers: Minimum number of workers
            max_workers: Maximum number of workers
            metrics_interval: How often to check metrics (seconds)
            scale_up_threshold: CPU threshold for scaling up (0-1)
            scale_down_threshold: CPU threshold for scaling down (0-1)
            cooldown: Minimum seconds between scaling actions
            metrics_provider: Optional function to provide custom metrics
        """
        self.pipeline = pipeline
        self.auto_scaler = AutoScaler(
            min_workers=min_workers,
            max_workers=max_workers,
            metrics_interval=metrics_interval,
            metrics_provider=metrics_provider
        )
        
        # Add default scaling rules
        self._add_default_rules(scale_up_threshold, scale_down_threshold, cooldown)
        
        # Prometheus metrics
        self.metrics = {
            'current_workers': Gauge('pipeline_workers', 'Current number of workers'),
            'scale_events': Counter('pipeline_scale_events', 'Number of scaling events', ['direction']),
            'queue_size': Gauge('pipeline_queue_size', 'Current queue size'),
            'processing_rate': Gauge('pipeline_processing_rate', 'Items processed per second')
        }
    
    def _add_default_rules(
        self,
        scale_up_threshold: float,
        scale_down_threshold: float,
        cooldown: float
    ) -> None:
        """Add default scaling rules."""
        # Scale up when CPU is high
        self.auto_scaler.add_rule(
            ScalingRule(
                metric=ScalingMetric.CPU_UTILIZATION,
                threshold=scale_up_threshold,
                operator='>',
                action=lambda x: min(x * 2, self.auto_scaler.max_workers),
                cooldown=cooldown
            )
        )
        
        # Scale down when CPU is low
        self.auto_scaler.add_rule(
            ScalingRule(
                metric=ScalingMetric.CPU_UTILIZATION,
                threshold=scale_down_threshold,
                operator='<',
                action=lambda x: max(x // 2, self.auto_scaler.min_workers),
                cooldown=cooldown
            )
        )
        
        # Scale based on queue length (custom metric)
        self.auto_scaler.add_rule(
            ScalingRule(
                metric=ScalingMetric.CUSTOM,
                metric_name='queue_length',
                threshold=1000,  # Max items in queue
                operator='>',
                action=lambda x: min(x + 2, self.auto_scaler.max_workers),
                cooldown=cooldown / 2  # More aggressive scaling for queue
            )
        )
    
    def start(self) -> None:
        """Start the auto-scaler and monitoring."""
        self.auto_scaler.start()
        logger.info("Pipeline scaler started")
    
    def stop(self) -> None:
        """Stop the auto-scaler and cleanup."""
        self.auto_scaler.stop()
        logger.info("Pipeline scaler stopped")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return {
            'workers': self.auto_scaler.current_workers,
            'min_workers': self.auto_scaler.min_workers,
            'max_workers': self.auto_scaler.max_workers,
            'rules': [{
                'metric': rule.metric.name,
                'threshold': rule.threshold,
                'operator': rule.operator,
                'cooldown': rule.cooldown
            } for rule in self.auto_scaler.rules]
        }
    
    def update_scaling_parameters(
        self,
        min_workers: Optional[int] = None,
        max_workers: Optional[int] = None,
        scale_up_threshold: Optional[float] = None,
        scale_down_threshold: Optional[float] = None,
        cooldown: Optional[float] = None
    ) -> None:
        """Update scaling parameters dynamically."""
        with self.auto_scaler.lock:
            if min_workers is not None:
                self.auto_scaler.min_workers = min_workers
            if max_workers is not None:
                self.auto_scaler.max_workers = max_workers
            
            # Update or add rules
            if any([scale_up_threshold, scale_down_threshold, cooldown]):
                self.auto_scaler.rules = []
                self._add_default_rules(
                    scale_up_threshold or 0.7,
                    scale_down_threshold or 0.3,
                    cooldown or 60.0
                )
        
        logger.info(f"Updated scaling parameters: {self.get_metrics()}")

# Example usage:
if __name__ == "__main__":
    import logging
    import random
    import time
    
    logging.basicConfig(level=logging.INFO)
    
    # Create a simple pipeline
    pipeline = StreamPipeline([
        # Your operators here
    ])
    
    # Create and start the scaler
    scaler = PipelineScaler(
        pipeline=pipeline,
        min_workers=2,
        max_workers=10,
        metrics_interval=5.0,
        scale_up_threshold=0.7,
        scale_down_threshold=0.3,
        cooldown=30.0
    )
    
    # Custom metrics provider
    def get_custom_metrics() -> Dict[str, float]:
        return {
            'queue_length': random.uniform(0, 2000),  # Simulate queue length
            'processing_latency': random.uniform(0, 500)  # ms
        }
    
    scaler.auto_scaler.metrics_provider = get_custom_metrics
    scaler.start()
    
    try:
        # Run for a while
        for _ in range(100):
            print(f"Current workers: {scaler.auto_scaler.current_workers}")
            time.sleep(5)
    except KeyboardInterrupt:
        pass
    finally:
        scaler.stop()
