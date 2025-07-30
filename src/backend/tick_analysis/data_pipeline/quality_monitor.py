"""
Data Quality Monitor Module

This module provides advanced data quality monitoring capabilities including
drift detection, anomaly detection, and comprehensive quality metrics.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import json
import os

logger = logging.getLogger(__name__)

@dataclass
class QualityMetric:
    """Represents a data quality metric with history."""
    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    threshold: Optional[float] = None
    severity: str = 'warning'  # 'info', 'warning', 'error', 'critical'
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DriftDetectionConfig:
    """Configuration for drift detection."""
    window_size: int = 1000
    significance_level: float = 0.05
    min_samples: int = 100
    features: List[str] = field(default_factory=list)
    methods: List[str] = field(default_factory=lambda: ['ks_test', 'wasserstein'])

class DataQualityMonitor:
    """Advanced data quality monitoring with drift detection and anomaly detection."""
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 storage_path: str = './data/quality_metrics'):
        """
        Initialize the data quality monitor.
        
        Args:
            config: Optional configuration dictionary
            storage_path: Path to store quality metrics
        """
        self.config = config or {}
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        
        # Initialize metrics storage
        self.metrics: Dict[str, List[QualityMetric]] = {}
        self.drift_detector = DriftDetector(DriftDetectionConfig(**self.config.get('drift_detection', {})))
        self.anomaly_detector = AnomalyDetector(self.config.get('anomaly_detection', {}))
        
        # Initialize reference data
        self.reference_data: Optional[pd.DataFrame] = None
        self.reference_stats: Dict[str, Dict[str, float]] = {}
    
    def set_reference_data(self, data: pd.DataFrame) -> None:
        """
        Set reference data for drift detection.
        
        Args:
            data: Reference dataset
        """
        self.reference_data = data
        self.reference_stats = self._calculate_statistics(data)
        logger.info("Reference data set for drift detection")
    
    def _calculate_statistics(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate statistical properties of the data."""
        stats = {}
        for column in data.select_dtypes(include=[np.number]).columns:
            stats[column] = {
                'mean': data[column].mean(),
                'std': data[column].std(),
                'min': data[column].min(),
                'max': data[column].max(),
                'skew': data[column].skew(),
                'kurtosis': data[column].kurtosis()
            }
        return stats
    
    async def check_quality(self, data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> List[QualityMetric]:
        """
        Perform comprehensive quality checks on the data.
        
        Args:
            data: Data to check (single record or batch)
            
        Returns:
            List of quality metrics
        """
        metrics = []
        
        # Convert to DataFrame if needed
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = pd.DataFrame(data)
        
        # Basic quality checks
        metrics.extend(self._check_basic_quality(df))
        
        # Drift detection if reference data is available
        if self.reference_data is not None:
            drift_metrics = self.drift_detector.detect_drift(df, self.reference_data)
            metrics.extend(drift_metrics)
        
        # Anomaly detection
        anomaly_metrics = self.anomaly_detector.detect_anomalies(df)
        metrics.extend(anomaly_metrics)
        
        # Store metrics
        for metric in metrics:
            self._store_metric(metric)
        
        return metrics
    
    def _check_basic_quality(self, df: pd.DataFrame) -> List[QualityMetric]:
        """Perform basic quality checks."""
        metrics = []
        
        # Check for missing values
        missing_pct = df.isnull().mean()
        for column, pct in missing_pct.items():
            metrics.append(QualityMetric(
                name=f'missing_values_{column}',
                value=float(pct),
                threshold=0.05,  # 5% missing values threshold
                severity='error' if pct > 0.05 else 'warning' if pct > 0.01 else 'info'
            ))
        
        # Check for duplicates
        duplicate_pct = df.duplicated().mean()
        metrics.append(QualityMetric(
            name='duplicate_records',
            value=float(duplicate_pct),
            threshold=0.01,  # 1% duplicates threshold
            severity='error' if duplicate_pct > 0.01 else 'warning' if duplicate_pct > 0.001 else 'info'
        ))
        
        # Check numeric ranges
        for column in df.select_dtypes(include=[np.number]).columns:
            if column in self.reference_stats:
                ref_stats = self.reference_stats[column]
                current_mean = df[column].mean()
                current_std = df[column].std()
                
                # Check mean drift
                z_score = abs(current_mean - ref_stats['mean']) / ref_stats['std']
                metrics.append(QualityMetric(
                    name=f'mean_drift_{column}',
                    value=float(z_score),
                    threshold=3.0,  # 3 standard deviations
                    severity='error' if z_score > 3.0 else 'warning' if z_score > 2.0 else 'info'
                ))
                
                # Check standard deviation drift
                std_ratio = current_std / ref_stats['std']
                metrics.append(QualityMetric(
                    name=f'std_drift_{column}',
                    value=float(std_ratio),
                    threshold=2.0,  # 2x standard deviation
                    severity='error' if std_ratio > 2.0 or std_ratio < 0.5 else 'warning' if std_ratio > 1.5 or std_ratio < 0.67 else 'info'
                ))
        
        return metrics
    
    def _store_metric(self, metric: QualityMetric) -> None:
        """Store a quality metric."""
        if metric.name not in self.metrics:
            self.metrics[metric.name] = []
        
        self.metrics[metric.name].append(metric)
        
        # Persist to disk
        self._persist_metrics()
    
    def _persist_metrics(self) -> None:
        """Persist metrics to disk."""
        try:
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            filepath = os.path.join(self.storage_path, f'quality_metrics_{timestamp}.json')
            
            # Convert metrics to serializable format
            data = {
                name: [
                    {
                        'value': m.value,
                        'timestamp': m.timestamp.isoformat(),
                        'threshold': m.threshold,
                        'severity': m.severity,
                        'metadata': m.metadata
                    }
                    for m in metrics
                ]
                for name, metrics in self.metrics.items()
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error persisting quality metrics: {e}")
    
    def get_metrics(self, 
                   metric_name: Optional[str] = None,
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None) -> Dict[str, List[QualityMetric]]:
        """
        Get quality metrics with optional filtering.
        
        Args:
            metric_name: Optional metric name to filter by
            start_time: Optional start time filter
            end_time: Optional end time filter
            
        Returns:
            Dictionary of metric name to list of metrics
        """
        if metric_name:
            metrics = {metric_name: self.metrics.get(metric_name, [])}
        else:
            metrics = self.metrics.copy()
        
        # Apply time filters
        if start_time or end_time:
            filtered_metrics = {}
            for name, metric_list in metrics.items():
                filtered = [
                    m for m in metric_list
                    if (not start_time or m.timestamp >= start_time) and
                       (not end_time or m.timestamp <= end_time)
                ]
                if filtered:
                    filtered_metrics[name] = filtered
            metrics = filtered_metrics
        
        return metrics

class DriftDetector:
    """Detects data drift using statistical tests."""
    
    def __init__(self, config: DriftDetectionConfig):
        """
        Initialize drift detector.
        
        Args:
            config: Drift detection configuration
        """
        self.config = config
    
    def detect_drift(self, 
                    current_data: pd.DataFrame,
                    reference_data: pd.DataFrame) -> List[QualityMetric]:
        """
        Detect drift between current and reference data.
        
        Args:
            current_data: Current data batch
            reference_data: Reference dataset
            
        Returns:
            List of drift metrics
        """
        metrics = []
        
        # Select numeric features
        features = self.config.features or current_data.select_dtypes(include=[np.number]).columns
        
        for feature in features:
            if feature not in current_data.columns or feature not in reference_data.columns:
                continue
            
            current = current_data[feature].dropna()
            reference = reference_data[feature].dropna()
            
            if len(current) < self.config.min_samples or len(reference) < self.config.min_samples:
                continue
            
            # Kolmogorov-Smirnov test
            if 'ks_test' in self.config.methods:
                ks_stat, p_value = stats.ks_2samp(current, reference)
                metrics.append(QualityMetric(
                    name=f'drift_ks_{feature}',
                    value=float(p_value),
                    threshold=self.config.significance_level,
                    severity='error' if p_value < self.config.significance_level else 'info',
                    metadata={'statistic': float(ks_stat)}
                ))
            
            # Wasserstein distance
            if 'wasserstein' in self.config.methods:
                wasserstein_dist = stats.wasserstein_distance(current, reference)
                metrics.append(QualityMetric(
                    name=f'drift_wasserstein_{feature}',
                    value=float(wasserstein_dist),
                    threshold=0.1,  # Arbitrary threshold, adjust based on domain
                    severity='error' if wasserstein_dist > 0.1 else 'warning' if wasserstein_dist > 0.05 else 'info'
                ))
        
        return metrics

class AnomalyDetector:
    """Detects anomalies in data using isolation forest."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize anomaly detector.
        
        Args:
            config: Anomaly detection configuration
        """
        self.config = config
        self.model = IsolationForest(
            contamination=config.get('contamination', 0.1),
            random_state=config.get('random_state', 42)
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def detect_anomalies(self, data: pd.DataFrame) -> List[QualityMetric]:
        """
        Detect anomalies in the data.
        
        Args:
            data: Data to check for anomalies
            
        Returns:
            List of anomaly metrics
        """
        metrics = []
        
        # Select numeric features
        features = data.select_dtypes(include=[np.number]).columns
        
        if len(features) == 0:
            return metrics
        
        # Prepare data
        X = data[features].dropna()
        
        if len(X) == 0:
            return metrics
        
        # Scale data
        if not self.is_fitted:
            self.scaler.fit(X)
            self.is_fitted = True
        
        X_scaled = self.scaler.transform(X)
        
        # Fit model if not already fitted
        if not hasattr(self.model, 'predict'):
            self.model.fit(X_scaled)
        
        # Predict anomalies
        predictions = self.model.predict(X_scaled)
        anomaly_ratio = (predictions == -1).mean()
        
        metrics.append(QualityMetric(
            name='anomaly_ratio',
            value=float(anomaly_ratio),
            threshold=self.config.get('anomaly_threshold', 0.1),
            severity='error' if anomaly_ratio > self.config.get('anomaly_threshold', 0.1) else 'warning' if anomaly_ratio > self.config.get('anomaly_threshold', 0.1) / 2 else 'info'
        ))
        
        return metrics 