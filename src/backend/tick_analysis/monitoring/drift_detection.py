"""
Drift Detection Module

Implements statistical and ML-based drift detection for monitoring data distribution shifts.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
from datetime import datetime
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing_extensions import Protocol
import warnings

logger = logging.getLogger(__name__)

class DriftType(Enum):
    """Types of data drift that can be detected."""
    COVARIATE_SHIFT = auto()
    CONCEPT_DRIFT = auto()
    DATA_DRIFT = auto()
    LABEL_DRIFT = auto()
    PREDICTION_DRIFT = auto()

class DriftSeverity(Enum):
    """Severity levels for detected drift."""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class DriftResult:
    """Container for drift detection results."""
    drift_detected: bool
    drift_type: DriftType
    severity: DriftSeverity
    statistic: Optional[float] = None
    p_value: Optional[float] = None
    threshold: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            'drift_detected': self.drift_detected,
            'drift_type': self.drift_type.name,
            'severity': self.severity.name,
            'statistic': self.statistic,
            'p_value': self.p_value,
            'threshold': self.threshold,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }

class DriftDetector(Protocol):
    """Protocol for drift detection implementations."""
    def fit(self, reference_data: Union[np.ndarray, pd.DataFrame]) -> None: ...
    def detect_drift(self, data: Union[np.ndarray, pd.DataFrame]) -> DriftResult: ...

class BaseDriftDetector:
    """Base class for drift detectors with common functionality."""
    
    def __init__(
        self,
        alpha: float = 0.05,
        window_size: int = 1000,
        min_samples: int = 100,
        **kwargs
    ):
        self.alpha = alpha
        self.window_size = window_size
        self.min_samples = min_samples
        self.is_fitted = False
        self.reference_stats: Dict[str, Any] = {}
    
    def _validate_input(self, data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Validate and convert input data."""
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        if not isinstance(data, np.ndarray):
            raise ValueError("Input must be numpy array or pandas DataFrame")
            
        if len(data) < self.min_samples:
            warnings.warn(
                f"Insufficient samples: {len(data)} < {self.min_samples}. "
                "Results may be unreliable."
            )
            
        return data
    
    def _calculate_severity(
        self,
        p_value: float,
        thresholds: Optional[Dict[DriftSeverity, float]] = None
    ) -> DriftSeverity:
        """Calculate drift severity based on p-value."""
        if thresholds is None:
            thresholds = {
                DriftSeverity.CRITICAL: 0.001,
                DriftSeverity.HIGH: 0.01,
                DriftSeverity.MEDIUM: 0.05,
                DriftSeverity.LOW: 0.1
            }
        
        if p_value < thresholds.get(DriftSeverity.CRITICAL, 0.001):
            return DriftSeverity.CRITICAL
        elif p_value < thresholds.get(DriftSeverity.HIGH, 0.01):
            return DriftSeverity.HIGH
        elif p_value < thresholds.get(DriftSeverity.MEDIUM, 0.05):
            return DriftSeverity.MEDIUM
        elif p_value < thresholds.get(DriftSeverity.LOW, 0.1):
            return DriftSeverity.LOW
        return DriftSeverity.NONE

class StatisticalDriftDetector(BaseDriftDetector):
    """Detects drift using statistical tests."""
    
    def __init__(
        self,
        test: str = 'ks',
        alpha: float = 0.05,
        **kwargs
    ):
        super().__init__(alpha=alpha, **kwargs)
        self.test = test
        self.reference_data: Optional[np.ndarray] = None
    
    def fit(self, reference_data: Union[np.ndarray, pd.DataFrame]) -> None:
        """Fit detector to reference data."""
        self.reference_data = self._validate_input(reference_data)
        self.reference_stats = {
            'mean': np.mean(self.reference_data, axis=0),
            'std': np.std(self.reference_data, axis=0),
            'min': np.min(self.reference_data, axis=0),
            'max': np.max(self.reference_data, axis=0)
        }
        self.is_fitted = True
    
    def detect_drift(self, data: Union[np.ndarray, pd.DataFrame]) -> DriftResult:
        """Detect drift in the given data compared to reference."""
        if not self.is_fitted:
            raise RuntimeError("Detector not fitted. Call fit() first.")
            
        data = self._validate_input(data)
        
        if self.test == 'ks':
            return self._ks_test(data)
        elif self.test == 't_test':
            return self._t_test(data)
        elif self.test == 'mannwhitneyu':
            return self._mannwhitneyu_test(data)
        else:
            raise ValueError(f"Unsupported test: {self.test}")
    
    def _ks_test(self, data: np.ndarray) -> DriftResult:
        """Perform Kolmogorov-Smirnov test for each feature."""
        p_values = []
        statistics = []
        
        for i in range(self.reference_data.shape[1]):
            ref = self.reference_data[:, i]
            current = data[:, i]
            
            if len(np.unique(ref)) == 1 and len(np.unique(current)) == 1:
                # Handle constant data
                stat, p_val = 0.0, 1.0
            else:
                stat, p_val = stats.ks_2samp(ref, current)
            
            p_values.append(p_val)
            statistics.append(stat)
        
        # Use min p-value with Bonferroni correction
        min_p = min(p_values) * len(p_values)
        min_p = min(1.0, max(0.0, min_p))  # Ensure p-value is in [0, 1]
        
        drift_detected = min_p < self.alpha
        severity = self._calculate_severity(min_p)
        
        return DriftResult(
            drift_detected=drift_detected,
            drift_type=DriftType.DATA_DRIFT,
            severity=severity,
            statistic=np.mean(statistics),
            p_value=float(min_p),
            threshold=self.alpha,
            metadata={
                'test': 'ks_2samp',
                'p_values': [float(p) for p in p_values],
                'statistics': [float(s) for s in statistics]
            }
        )
    
    def _t_test(self, data: np.ndarray) -> DriftResult:
        """Perform independent t-test for each feature."""
        p_values = []
        statistics = []
        
        for i in range(self.reference_data.shape[1]):
            ref = self.reference_data[:, i]
            current = data[:, i]
            
            if len(np.unique(np.concatenate([ref, current]))) == 1:
                # Handle constant data
                stat, p_val = 0.0, 1.0
            else:
                stat, p_val = stats.ttest_ind(ref, current, equal_var=False)
            
            p_values.append(p_val)
            statistics.append(stat)
        
        # Use min p-value with Bonferroni correction
        min_p = min(p_values) * len(p_values)
        min_p = min(1.0, max(0.0, min_p))
        
        drift_detected = min_p < self.alpha
        severity = self._calculate_severity(min_p)
        
        return DriftResult(
            drift_detected=drift_detected,
            drift_type=DriftType.COVARIATE_SHIFT,
            severity=severity,
            statistic=np.mean(statistics),
            p_value=float(min_p),
            threshold=self.alpha,
            metadata={
                'test': 'ttest_ind',
                'p_values': [float(p) for p in p_values],
                'statistics': [float(s) for s in statistics]
            }
        )
    
    def _mannwhitneyu_test(self, data: np.ndarray) -> DriftResult:
        """Perform Mann-Whitney U test for each feature."""
        p_values = []
        statistics = []
        
        for i in range(self.reference_data.shape[1]):
            ref = self.reference_data[:, i]
            current = data[:, i]
            
            if len(np.unique(np.concatenate([ref, current]))) == 1:
                # Handle constant data
                stat, p_val = 0.0, 1.0
            else:
                stat, p_val = stats.mannwhitneyu(ref, current, alternative='two-sided')
            
            p_values.append(p_val)
            statistics.append(stat)
        
        # Use min p-value with Bonferroni correction
        min_p = min(p_values) * len(p_values)
        min_p = min(1.0, max(0.0, min_p))
        
        drift_detected = min_p < self.alpha
        severity = self._calculate_severity(min_p)
        
        return DriftResult(
            drift_detected=drift_detected,
            drift_type=DriftType.DATA_DRIFT,
            severity=severity,
            statistic=np.mean(statistics),
            p_value=float(min_p),
            threshold=self.alpha,
            metadata={
                'test': 'mannwhitneyu',
                'p_values': [float(p) for p in p_values],
                'statistics': [float(s) for s in statistics]
            }
        )

class MLDriftDetector(BaseDriftDetector):
    """Detects drift using machine learning models."""
    
    def __init__(
        self,
        model: str = 'isolation_forest',
        contamination: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model_type = model
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.model: Any = None
    
    def fit(self, reference_data: Union[np.ndarray, pd.DataFrame]) -> None:
        """Fit detector to reference data."""
        reference_data = self._validate_input(reference_data)
        
        # Scale the data
        self.reference_data = self.scaler.fit_transform(reference_data)
        
        # Initialize and fit the model
        if self.model_type == 'isolation_forest':
            self.model = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unsupported model: {self.model_type}")
        
        self.model.fit(self.reference_data)
        self.is_fitted = True
    
    def detect_drift(self, data: Union[np.ndarray, pd.DataFrame]) -> DriftResult:
        """Detect drift in the given data compared to reference."""
        if not self.is_fitted:
            raise RuntimeError("Detector not fitted. Call fit() first.")
        
        data = self._validate_input(data)
        
        # Scale the data
        data_scaled = self.scaler.transform(data)
        
        # Get anomaly scores (lower means more anomalous)
        scores = -self.model.score_samples(data_scaled)
        
        # Calculate drift metrics
        threshold = np.percentile(scores, 100 * (1 - self.contamination))
        drift_fraction = np.mean(scores > threshold)
        
        # Calculate p-value using binomial test
        n = len(scores)
        k = np.sum(scores > threshold)
        p_value = stats.binomtest(
            k=k,
            n=n,
            p=self.contamination,
            alternative='greater'
        ).pvalue
        
        drift_detected = p_value < self.alpha
        severity = self._calculate_severity(p_value)
        
        return DriftResult(
            drift_detected=drift_detected,
            drift_type=DriftType.DATA_DRIFT,
            severity=severity,
            statistic=float(drift_fraction),
            p_value=float(p_value),
            threshold=float(threshold),
            metadata={
                'model': self.model_type,
                'anomaly_scores': scores.tolist(),
                'threshold': float(threshold),
                'contamination': self.contamination
            }
        )

class DriftDetectorFactory:
    """Factory for creating drift detector instances."""
    
    @staticmethod
    def create_detector(
        detector_type: str = 'statistical',
        detector_params: Optional[Dict[str, Any]] = None
    ) -> DriftDetector:
        """Create a drift detector of the specified type."""
        detector_params = detector_params or {}
        
        if detector_type == 'statistical':
            return StatisticalDriftDetector(**detector_params)
        elif detector_type == 'ml':
            return MLDriftDetector(**detector_params)
        else:
            raise ValueError(f"Unknown detector type: {detector_type}")

class DriftMonitor:
    """Monitors data streams for drift over time."""
    
    def __init__(
        self,
        reference_data: Union[np.ndarray, pd.DataFrame],
        detector_type: str = 'statistical',
        detector_params: Optional[Dict[str, Any]] = None,
        check_interval: int = 1000,
        warmup_period: int = 1000
    ):
        """Initialize the drift monitor."""
        self.detector = DriftDetectorFactory.create_detector(
            detector_type=detector_type,
            detector_params=detector_params or {}
        )
        
        self.detector.fit(reference_data)
        self.check_interval = check_interval
        self.warmup_period = warmup_period
        self.sample_count = 0
        self.buffer: List[np.ndarray] = []
        self.drift_history: List[DriftResult] = []
    
    def update(self, sample: Union[np.ndarray, pd.Series, Dict[str, Any]]) -> Optional[DriftResult]:
        """Update the monitor with a new sample."""
        # Convert sample to numpy array
        if isinstance(sample, dict):
            sample = np.array(list(sample.values()))
        elif isinstance(sample, pd.Series):
            sample = sample.values
        
        if not isinstance(sample, np.ndarray):
            sample = np.array([sample])
        
        # Ensure 2D array
        if len(sample.shape) == 1:
            sample = sample.reshape(1, -1)
        
        # Add to buffer
        self.buffer.append(sample)
        self.sample_count += sample.shape[0]
        
        # Check for drift if we have enough samples
        if self.sample_count >= self.warmup_period and len(self.buffer) >= self.check_interval:
            # Concatenate buffered samples
            window = np.concatenate(self.buffer, axis=0)
            
            # Reset buffer
            self.buffer = []
            
            # Check for drift
            result = self.detector.detect_drift(window)
            self.drift_history.append(result)
            
            return result
        
        return None
    
    def get_drift_history(self) -> List[Dict[str, Any]]:
        """Get the history of drift detection results."""
        return [r.to_dict() for r in self.drift_history]
    
    def reset(self) -> None:
        """Reset the monitor's state."""
        self.sample_count = 0
        self.buffer = []
        self.drift_history = []
