"""Auto-scaling manager for dynamic resource allocation."""
from typing import Dict, List, Optional, Callable, Any
import time
import logging
import threading
from dataclasses import dataclass
from enum import Enum, auto
from concurrent.futures import ThreadPoolExecutor, Executor
import numpy as np
from prometheus_client import Gauge, Histogram

logger = logging.getLogger(__name__)

class ScalingMetric(Enum):
    CPU_UTILIZATION = auto()
    MEMORY_USAGE = auto()
    QUEUE_LENGTH = auto()
    LATENCY = auto()
    CUSTOM = auto()

@dataclass
class ScalingRule:
    metric: ScalingMetric
    threshold: float
    operator: str  # '>', '<', '>=', '<=', '==', '!='
    action: Callable[[int], int]  # Function to adjust worker count
    cooldown: float = 60.0  # Seconds to wait between scaling actions
    metric_name: Optional[str] = None  # For CUSTOM metrics
    last_action_time: float = 0.0

class AutoScaler:
    """Manages dynamic scaling of worker pools based on metrics."""
    
    def __init__(
        self,
        min_workers: int = 1,
        max_workers: int = 100,
        metrics_interval: float = 5.0,
        metrics_provider: Optional[Callable[[], Dict[str, float]]] = None
    ):
        """Initialize the auto-scaler.
        
        Args:
            min_workers: Minimum number of workers
            max_workers: Maximum number of workers
            metrics_interval: How often to check metrics (seconds)
            metrics_provider: Optional function to provide custom metrics
        """
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.current_workers = min_workers
        self.metrics_interval = metrics_interval
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.RLock()
        self.rules: List[ScalingRule] = []
        self.metrics_provider = metrics_provider or {}
        self.executor: Optional[Executor] = None
        
        # Prometheus metrics
        self.worker_gauge = Gauge('autoscaler_workers', 'Current number of workers')
        self.scale_events = Gauge('autoscaler_scale_events', 'Number of scaling events', ['direction'])
        self.evaluation_duration = Histogram(
            'autoscaler_evaluation_seconds',
            'Time taken to evaluate scaling rules'
        )
    
    def add_rule(self, rule: ScalingRule) -> None:
        """Add a scaling rule."""
        with self.lock:
            self.rules.append(rule)
    
    def start(self) -> None:
        """Start the auto-scaler."""
        if self.running:
            logger.warning("Auto-scaler is already running")
            return
            
        self.running = True
        self.thread = threading.Thread(
            target=self._run,
            name="AutoScaler",
            daemon=True
        )
        self.thread.start()
        logger.info("Auto-scaler started")
    
    def stop(self) -> None:
        """Stop the auto-scaler."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=self.metrics_interval * 2)
        logger.info("Auto-scaler stopped")
    
    def _run(self) -> None:
        """Main monitoring loop."""
        while self.running:
            try:
                self._evaluate_rules()
            except Exception as e:
                logger.error(f"Error in auto-scaler: {e}", exc_info=True)
            
            # Sleep until next evaluation
            time.sleep(self.metrics_interval)
    
    def _evaluate_rules(self) -> None:
        """Evaluate all scaling rules and take action if needed."""
        with self.evaluation_duration.time():
            metrics = self._collect_metrics()
            
            for rule in self.rules:
                current_time = time.time()
                
                # Check cooldown
                if current_time - rule.last_action_time < rule.cooldown:
                    continue
                
                # Get metric value
                if rule.metric == ScalingMetric.CUSTOM and rule.metric_name:
                    value = metrics.get(rule.metric_name, 0)
                else:
                    value = metrics.get(rule.metric.name.lower(), 0)
                
                # Check condition
                condition_met = False
                if rule.operator == '>':
                    condition_met = value > rule.threshold
                elif rule.operator == '>=':
                    condition_met = value >= rule.threshold
                elif rule.operator == '<':
                    condition_met = value < rule.threshold
                elif rule.operator == '<=':
                    condition_met = value <= rule.threshold
                elif rule.operator == '==':
                    condition_met = value == rule.threshold
                elif rule.operator == '!=':
                    condition_met = value != rule.threshold
                
                if condition_met:
                    self._execute_rule(rule, current_time)
    
    def _execute_rule(self, rule: ScalingRule, timestamp: float) -> None:
        """Execute a scaling rule."""
        with self.lock:
            new_count = rule.action(self.current_workers)
            new_count = max(self.min_workers, min(self.max_workers, new_count))
            
            if new_count != self.current_workers:
                old_count = self.current_workers
                self.current_workers = new_count
                rule.last_action_time = timestamp
                
                # Update executor if configured
                if self.executor and hasattr(self.executor, '_max_workers'):
                    self.executor._max_workers = new_count
                
                # Update metrics
                self.worker_gauge.set(new_count)
                direction = 'up' if new_count > old_count else 'down'
                self.scale_events.labels(direction=direction).inc()
                
                logger.info(
                    f"Scaling workers: {old_count} -> {new_count} "
                    f"(rule: {rule.metric.name} {rule.operator} {rule.threshold})"
                )
    
    def _collect_metrics(self) -> Dict[str, float]:
        """Collect system and application metrics."""
        metrics = {}
        
        try:
            # System metrics
            import psutil
            process = psutil.Process()
            
            # CPU utilization (percentage)
            metrics['cpu_utilization'] = psutil.cpu_percent(interval=0.1)
            
            # Memory usage (percentage)
            metrics['memory_usage'] = process.memory_percent()
            
            # Thread count
            metrics['thread_count'] = process.num_threads()
            
            # Custom metrics from provider
            if callable(self.metrics_provider):
                try:
                    custom_metrics = self.metrics_provider()
                    if isinstance(custom_metrics, dict):
                        metrics.update(custom_metrics)
                except Exception as e:
                    logger.error(f"Error getting custom metrics: {e}", exc_info=True)
                    
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}", exc_info=True)
        
        return metrics
    
    def get_executor(self) -> Executor:
        """Get a ThreadPoolExecutor that auto-scales with this scaler."""
        if not self.executor:
            self.executor = ThreadPoolExecutor(
                max_workers=self.current_workers,
                thread_name_prefix="scaling_worker"
            )
        return self.executor
