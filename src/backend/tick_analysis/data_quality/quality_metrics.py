"""
Data quality metrics and monitoring.
"""

from typing import Dict, List, Optional, Any, Union, Tuple, Callable
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
from datetime import datetime, timedelta
from scipy import stats
import json

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of data quality metrics."""
    COMPLETENESS = auto()
    ACCURACY = auto()
    CONSISTENCY = auto()
    TIMELINESS = auto()
    VALIDITY = auto()
    UNIQUENESS = auto()

class ThresholdType(Enum):
    """Types of thresholds for metrics."""
    LOWER_BOUND = auto()
    UPPER_BOUND = auto()
    RANGE = auto()
    CATEGORICAL = auto()

@dataclass
class MetricThreshold:
    """Threshold configuration for a metric."""
    threshold_type: ThresholdType
    warning: Optional[float] = None
    error: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    categories: Optional[List[Any]] = None
    
    def check(self, value: float) -> Tuple[bool, str]:
        """
        Check if a value meets the threshold.
        
        Returns:
            Tuple of (is_ok, message)
        """
        if self.threshold_type == ThresholdType.LOWER_BOUND:
            if self.error is not None and value < self.error:
                return False, f"Value {value} below error threshold {self.error}"
            if self.warning is not None and value < self.warning:
                return True, f"Value {value} below warning threshold {self.warning}"
            return True, ""
            
        elif self.threshold_type == ThresholdType.UPPER_BOUND:
            if self.error is not None and value > self.error:
                return False, f"Value {value} above error threshold {self.error}"
            if self.warning is not None and value > self.warning:
                return True, f"Value {value} above warning threshold {self.warning}"
            return True, ""
            
        elif self.threshold_type == ThresholdType.RANGE:
            if (self.min_value is not None and value < self.min_value) or \
               (self.max_value is not None and value > self.max_value):
                return False, f"Value {value} outside range [{self.min_value}, {self.max_value}]"
            return True, ""
            
        elif self.threshold_type == ThresholdType.CATEGORICAL:
            if self.categories is not None and value not in self.categories:
                return False, f"Value {value} not in allowed categories {self.categories}"
            return True, ""
            
        return True, ""

@dataclass
class QualityMetric:
    """Configuration for a data quality metric."""
    name: str
    metric_type: MetricType
    description: str = ""
    column: Optional[str] = None
    threshold: Optional[MetricThreshold] = None
    compute_function: Optional[Callable] = None
    required: bool = True
    weight: float = 1.0
    
    def compute(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute the metric value for a DataFrame.
        
        Returns:
            Dictionary with metric results
        """
        if self.compute_function is not None:
            return self.compute_function(df)
            
        if self.column is None or self.column not in df.columns:
            return {
                'value': None,
                'valid': not self.required,
                'message': f"Column '{self.column}' not found" if self.column else "No column specified",
                'metric_type': self.metric_type.name,
                'threshold': None
            }
        
        try:
            if self.metric_type == MetricType.COMPLETENESS:
                return self._compute_completeness(df)
            elif self.metric_type == MetricType.ACCURACY:
                return self._compute_accuracy(df)
            elif self.metric_type == MetricType.CONSISTENCY:
                return self._compute_consistency(df)
            elif self.metric_type == MetricType.TIMELINESS:
                return self._compute_timeliness(df)
            elif self.metric_type == MetricType.VALIDITY:
                return self._compute_validity(df)
            elif self.metric_type == MetricType.UNIQUENESS:
                return self._compute_uniqueness(df)
            else:
                return {
                    'value': None,
                    'valid': True,
                    'message': f"Unsupported metric type: {self.metric_type}",
                    'metric_type': self.metric_type.name,
                    'threshold': None
                }
        except Exception as e:
            return {
                'value': None,
                'valid': False,
                'message': f"Error computing metric: {str(e)}",
                'metric_type': self.metric_type.name,
                'threshold': None
            }
    
    def _compute_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute completeness metric (non-null ratio)."""
        non_null_count = df[self.column].count()
        total = len(df)
        ratio = non_null_count / total if total > 0 else 0.0
        
        result = {
            'value': ratio,
            'valid': True,
            'message': "",
            'metric_type': self.metric_type.name,
            'details': {
                'non_null_count': int(non_null_count),
                'total_count': total,
                'null_count': total - non_null_count,
                'ratio': ratio
            }
        }
        
        if self.threshold:
            is_ok, message = self.threshold.check(ratio)
            result['valid'] = is_ok
            result['message'] = message
            result['threshold'] = {
                'type': self.threshold.threshold_type.name,
                'warning': self.threshold.warning,
                'error': self.threshold.error
            }
        
        return result
    
    def _compute_accuracy(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute accuracy metric (requires reference values)."""
        # This is a placeholder - in practice, you'd compare to ground truth
        return {
            'value': None,
            'valid': True,
            'message': "Accuracy computation requires reference values",
            'metric_type': self.metric_type.name
        }
    
    def _compute_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute consistency metric (value distribution)."""
        value_counts = df[self.column].value_counts(normalize=True)
        entropy = stats.entropy(value_counts)
        
        return {
            'value': float(entropy),
            'valid': True,
            'message': "",
            'metric_type': self.metric_type.name,
            'details': {
                'value_counts': value_counts.to_dict(),
                'entropy': float(entropy),
                'n_unique': len(value_counts)
            }
        }
    
    def _compute_timeliness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute timeliness metric (time since last update)."""
        if not pd.api.types.is_datetime64_any_dtype(df[self.column]):
            try:
                df[self.column] = pd.to_datetime(df[self.column])
            except:
                return {
                    'value': None,
                    'valid': False,
                    'message': f"Column '{self.column}' is not a valid datetime",
                    'metric_type': self.metric_type.name
                }
        
        now = pd.Timestamp.now()
        time_deltas = (now - df[self.column]).dt.total_seconds() / 3600  # Hours
        
        stats = {
            'min': float(time_deltas.min()),
            'max': float(time_deltas.max()),
            'mean': float(time_deltas.mean()),
            'median': float(time_deltas.median()),
            'std': float(time_deltas.std())
        }
        
        result = {
            'value': stats['mean'],
            'valid': True,
            'message': "",
            'metric_type': self.metric_type.name,
            'details': stats
        }
        
        if self.threshold:
            is_ok, message = self.threshold.check(stats['max'])
            result['valid'] = is_ok
            result['message'] = message
            result['threshold'] = {
                'type': self.threshold.threshold_type.name,
                'warning': self.threshold.warning,
                'error': self.threshold.error
            }
        
        return result
    
    def _compute_validity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute validity metric (adherence to constraints)."""
        # This is a placeholder - in practice, you'd validate against constraints
        return {
            'value': None,
            'valid': True,
            'message': "Validity computation requires constraints definition",
            'metric_type': self.metric_type.name
        }
    
    def _compute_uniqueness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute uniqueness metric (duplicate ratio)."""
        unique_count = df[self.column].nunique()
        total = len(df)
        ratio = unique_count / total if total > 0 else 0.0
        
        result = {
            'value': ratio,
            'valid': True,
            'message': "",
            'metric_type': self.metric_type.name,
            'details': {
                'unique_count': int(unique_count),
                'total_count': total,
                'duplicate_count': total - unique_count,
                'ratio': ratio
            }
        }
        
        if self.threshold:
            is_ok, message = self.threshold.check(1.0 - ratio)  # Duplicate ratio
            result['valid'] = is_ok
            result['message'] = message
            result['threshold'] = {
                'type': self.threshold.threshold_type.name,
                'warning': self.threshold.warning,
                'error': self.threshold.error
            }
        
        return result

class QualityMetrics:
    """
    Data quality assessment and monitoring.
    
    Features:
    - Define and compute data quality metrics
    - Track metrics over time
    - Generate quality reports
    - Set up alerts for quality issues
    """
    
    def __init__(self, metrics: Optional[List[QualityMetric]] = None):
        """
        Initialize with a list of quality metrics.
        
        Args:
            metrics: List of QualityMetric instances
        """
        self.metrics = {m.name: m for m in (metrics or [])}
        self.history = []
    
    def add_metric(self, metric: QualityMetric) -> None:
        """Add a quality metric."""
        self.metrics[metric.name] = metric
    
    def remove_metric(self, name: str) -> None:
        """Remove a quality metric by name."""
        if name in self.metrics:
            del self.metrics[name]
    
    def compute_metrics(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Compute all metrics for a DataFrame.
        
        Returns:
            Dictionary mapping metric names to results
        """
        results = {}
        for name, metric in self.metrics.items():
            try:
                results[name] = metric.compute(df)
            except Exception as e:
                logger.error(f"Error computing metric {name}: {str(e)}")
                results[name] = {
                    'value': None,
                    'valid': False,
                    'message': f"Error: {str(e)}",
                    'metric_type': 'ERROR'
                }
        
        # Record in history
        self.history.append({
            'timestamp': datetime.utcnow().isoformat(),
            'results': results
        })
        
        return results
    
    def get_metric_history(self, name: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get historical values for a metric.
        
        Args:
            name: Metric name
            limit: Maximum number of historical values to return
            
        Returns:
            List of historical metric values with timestamps
        """
        history = []
        for entry in reversed(self.history):
            if name in entry['results']:
                history.append({
                    'timestamp': entry['timestamp'],
                    **entry['results'][name]
                })
                if len(history) >= limit:
                    break
        return history
    
    def get_quality_score(self, results: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Compute an overall quality score from metric results.
        
        Args:
            results: Optional pre-computed metric results
            
        Returns:
            Dictionary with quality score and breakdown
        """
        if results is None:
            results = {}
        
        total_weight = 0.0
        weighted_sum = 0.0
        passed = 0
        failed = 0
        
        metrics_summary = {}
        
        for name, metric in self.metrics.items():
            if name not in results:
                continue
                
            result = results[name]
            weight = metric.weight
            
            # Skip metrics that don't have a computable value
            if result.get('value') is None and 'ratio' not in result.get('details', {}):
                continue
                
            # Get the primary value to use for scoring
            value = result.get('value')
            if value is None and 'ratio' in result.get('details', {}):
                value = result['details']['ratio']
                
            if value is None:
                continue
                
            # For metrics where higher is better (completeness, uniqueness)
            if metric.metric_type in [MetricType.COMPLETENESS, MetricType.UNIQUENESS]:
                score = value
            # For metrics where lower is better (timeliness max_delay)
            elif metric.metric_type == MetricType.TIMELINESS and 'max' in result.get('details', {}):
                # Normalize based on threshold if available
                if metric.threshold and metric.threshold.error is not None:
                    score = max(0, 1.0 - (result['details']['max'] / metric.threshold.error))
                else:
                    score = 1.0 / (1.0 + result['details']['max'] / 24.0)  # Normalize by days
            else:
                # Default: use validity
                score = 1.0 if result.get('valid', True) else 0.0
            
            # Apply weight
            weighted_sum += score * weight
            total_weight += weight
            
            # Track pass/fail
            if result.get('valid', True):
                passed += 1
            else:
                failed += 1
            
            metrics_summary[name] = {
                'score': score,
                'weight': weight,
                'value': value,
                'valid': result.get('valid', True),
                'message': result.get('message', '')
            }
        
        # Calculate overall score
        overall_score = (weighted_sum / total_weight) if total_weight > 0 else 1.0
        
        return {
            'score': overall_score,
            'passed': passed,
            'failed': failed,
            'total': passed + failed,
            'metrics': metrics_summary,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def generate_report(
        self, 
        df: Optional[pd.DataFrame] = None,
        results: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Generate a data quality report.
        
        Args:
            df: Optional DataFrame to compute metrics for
            results: Optional pre-computed metric results
            
        Returns:
            Dictionary with quality report
        """
        if results is None and df is not None:
            results = self.compute_metrics(df)
        elif results is None:
            results = {}
        
        # Get quality score
        quality_score = self.get_quality_score(results)
        
        # Categorize metrics by type
        metrics_by_type = {}
        for name, metric in self.metrics.items():
            if name not in results:
                continue
                
            metric_type = metric.metric_type.name.lower()
            if metric_type not in metrics_by_type:
                metrics_by_type[metric_type] = []
                
            result = results[name].copy()
            result['name'] = name
            result['description'] = metric.description
            result['column'] = metric.column
            result['required'] = metric.required
            result['weight'] = metric.weight
            
            metrics_by_type[metric_type].append(result)
        
        # Get recent history
        recent_history = []
        if self.history:
            for entry in reversed(self.history[-5:]):  # Last 5 entries
                score = self.get_quality_score(entry['results'])
                recent_history.append({
                    'timestamp': entry['timestamp'],
                    'score': score['score'],
                    'passed': score['passed'],
                    'failed': score['failed']
                })
        
        return {
            'summary': {
                'overall_score': quality_score['score'],
                'metrics_passed': quality_score['passed'],
                'metrics_failed': quality_score['failed'],
                'total_metrics': quality_score['total'],
                'timestamp': datetime.utcnow().isoformat(),
                'dataset_shape': {
                    'rows': len(df) if df is not None else 0,
                    'columns': len(df.columns) if df is not None else 0
                } if df is not None else None
            },
            'metrics_by_type': metrics_by_type,
            'recent_history': recent_history,
            'failed_metrics': [
                {'name': name, **result}
                for name, result in results.items()
                if not result.get('valid', True)
            ],
            'metric_details': quality_score['metrics']
        }
    
    def to_json(self) -> str:
        """Serialize metrics configuration to JSON."""
        config = {
            'metrics': [
                {
                    'name': m.name,
                    'metric_type': m.metric_type.name,
                    'description': m.description,
                    'column': m.column,
                    'required': m.required,
                    'weight': m.weight,
                    'threshold': {
                        'threshold_type': m.threshold.threshold_type.name,
                        'warning': m.threshold.warning,
                        'error': m.threshold.error,
                        'min_value': m.threshold.min_value,
                        'max_value': m.threshold.max_value,
                        'categories': m.threshold.categories
                    } if m.threshold else None
                }
                for m in self.metrics.values()
            ]
        }
        return json.dumps(config, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'QualityMetrics':
        """Create QualityMetrics instance from JSON configuration."""
        config = json.loads(json_str)
        metrics = []
        
        for m in config.get('metrics', []):
            threshold = None
            if m.get('threshold'):
                threshold = MetricThreshold(
                    threshold_type=ThresholdType[m['threshold']['threshold_type']],
                    warning=m['threshold'].get('warning'),
                    error=m['threshold'].get('error'),
                    min_value=m['threshold'].get('min_value'),
                    max_value=m['threshold'].get('max_value'),
                    categories=m['threshold'].get('categories')
                )
            
            metric = QualityMetric(
                name=m['name'],
                metric_type=MetricType[m['metric_type']],
                description=m.get('description', ''),
                column=m.get('column'),
                threshold=threshold,
                required=m.get('required', True),
                weight=m.get('weight', 1.0)
            )
            metrics.append(metric)
        
        return cls(metrics=metrics)
    
    @classmethod
    def get_default_metrics(cls, df: pd.DataFrame) -> 'QualityMetrics':
        """
        Create a default set of metrics based on DataFrame columns.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            QualityMetrics instance with default metrics
        """
        metrics = []
        
        for col in df.columns:
            # Skip columns that look like IDs or indices
            if any(x in col.lower() for x in ['id', 'idx', 'index', 'key', 'uuid']):
                continue
                
            # Completeness check for all columns
            metrics.append(QualityMetric(
                name=f"{col}_completeness",
                metric_type=MetricType.COMPLETENESS,
                description=f"Completeness of {col}",
                column=col,
                threshold=MetricThreshold(
                    threshold_type=ThresholdType.LOWER_BOUND,
                    warning=0.95,  # Warn if < 95% complete
                    error=0.90     # Error if < 90% complete
                ),
                required=True,
                weight=1.0
            ))
            
            # Uniqueness check for non-numeric columns
            if not pd.api.types.is_numeric_dtype(df[col]):
                metrics.append(QualityMetric(
                    name=f"{col}_uniqueness",
                    metric_type=MetricType.UNIQUENESS,
                    description=f"Uniqueness of {col} values",
                    column=col,
                    threshold=MetricThreshold(
                        threshold_type=ThresholdType.LOWER_BOUND,
                        warning=0.95,  # Warn if > 5% duplicates
                        error=0.90     # Error if > 10% duplicates
                    ),
                    required=False,
                    weight=0.5
                ))
            
            # Timeliness check for datetime columns
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                metrics.append(QualityMetric(
                    name=f"{col}_timeliness",
                    metric_type=MetricType.TIMELINESS,
                    description=f"Timeliness of {col}",
                    column=col,
                    threshold=MetricThreshold(
                        threshold_type=ThresholdType.UPPER_BOUND,
                        warning=24.0,  # Warn if data older than 24 hours
                        error=72.0     # Error if older than 72 hours
                    ),
                    required=False,
                    weight=0.8
                ))
        
        return cls(metrics=metrics)
