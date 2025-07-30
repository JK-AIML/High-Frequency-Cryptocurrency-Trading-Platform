"""
Data drift detection and monitoring system.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
from datetime import datetime, timedelta
import warnings
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class DriftType(Enum):
    """Types of data drift."""
    COVARIATE_DRIFT = auto()
    CONCEPT_DRIFT = auto()
    LABEL_DRIFT = auto()
    DATA_QUALITY_ISSUE = auto()

class DetectionMethod(Enum):
    """Drift detection methods."""
    STATISTICAL_TEST = auto()
    MMD = auto()  # Maximum Mean Discrepancy
    KL_DIVERGENCE = auto()
    JENSEN_SHANNON = auto()
    CHI_SQUARED = auto()
    KOLMOGOROV_SMIRNOV = auto()
    ISOLATION_FOREST = auto()
    MAHALANOBIS_DISTANCE = auto()

@dataclass
class DriftResult:
    """Result of drift detection."""
    has_drift: bool
    drift_type: DriftType
    detection_method: DetectionMethod
    p_value: Optional[float] = None
    test_statistic: Optional[float] = None
    threshold: Optional[float] = None
    confidence: float = 0.95
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'has_drift': self.has_drift,
            'drift_type': self.drift_type.name,
            'detection_method': self.detection_method.name,
            'p_value': self.p_value,
            'test_statistic': self.test_statistic,
            'threshold': self.threshold,
            'confidence': self.confidence,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp.isoformat()
        }

class DriftDetector:
    """
    Detects and monitors data drift between reference and current datasets.
    
    Supports:
    - Univariate drift detection
    - Multivariate drift detection
    - Concept drift detection
    - Custom drift detectors
    """
    
    def __init__(
        self,
        reference_data: Union[pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None,
        categorical_features: Optional[List[Union[str, int]]] = None,
        numerical_features: Optional[List[Union[str, int]]] = None,
        alpha: float = 0.05,
        random_state: Optional[int] = None
    ):
        """
        Initialize the drift detector.
        
        Args:
            reference_data: Reference dataset (baseline)
            feature_names: List of feature names
            categorical_features: List of categorical feature names/indices
            numerical_features: List of numerical feature names/indices
            alpha: Significance level for statistical tests
            random_state: Random seed for reproducibility
        """
        self.reference_data = self._ensure_dataframe(reference_data, feature_names)
        self.alpha = alpha
        self.random_state = random_state
        
        # Infer feature types if not provided
        if categorical_features is None and numerical_features is None:
            self.categorical_features, self.numerical_features = self._infer_feature_types()
        else:
            self.categorical_features = categorical_features or []
            self.numerical_features = numerical_features or []
        
        # Preprocess reference data
        self.scaler = StandardScaler()
        self._fit_reference()
    
    def _ensure_dataframe(
        self, 
        data: Union[pd.DataFrame, np.ndarray], 
        feature_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Convert input to DataFrame if it isn't already."""
        if isinstance(data, pd.DataFrame):
            return data.copy()
        elif isinstance(data, np.ndarray):
            if feature_names is None:
                feature_names = [f'feature_{i}' for i in range(data.shape[1])]
            return pd.DataFrame(data, columns=feature_names)
        else:
            raise ValueError("Input must be a pandas DataFrame or numpy array")
    
    def _infer_feature_types(self) -> Tuple[List[str], List[str]]:
        """Infer categorical and numerical features."""
        categorical = []
        numerical = []
        
        for col in self.reference_data.columns:
            if pd.api.types.is_numeric_dtype(self.reference_data[col]):
                # Check if it's actually categorical with numeric codes
                unique_ratio = self.reference_data[col].nunique() / len(self.reference_data[col])
                if unique_ratio < 0.05:  # Arbitrary threshold
                    categorical.append(col)
                else:
                    numerical.append(col)
            else:
                categorical.append(col)
        
        return categorical, numerical
    
    def _fit_reference(self) -> None:
        """Fit reference data statistics."""
        # Store reference statistics
        self.reference_stats = {}
        
        # Numerical features
        if self.numerical_features:
            num_data = self.reference_data[self.numerical_features]
            self.scaler.fit(num_data)
            
            # Store reference statistics
            self.reference_stats['mean'] = num_data.mean().to_dict()
            self.reference_stats['std'] = num_data.std().to_dict()
            self.reference_stats['percentiles'] = {
                col: np.percentile(num_data[col], [5, 25, 50, 75, 95])
                for col in num_data.columns
            }
        
        # Categorical features
        if self.categorical_features:
            cat_data = self.reference_data[self.categorical_features]
            self.reference_stats['categorical'] = {}
            
            for col in cat_data.columns:
                value_counts = cat_data[col].value_counts(normalize=True)
                self.reference_stats['categorical'][col] = {
                    'distribution': value_counts.to_dict(),
                    'n_categories': len(value_counts)
                }
        
        # Fit anomaly detectors
        if len(self.numerical_features) > 0:
            self._fit_anomaly_detectors()
    
    def _fit_anomaly_detectors(self) -> None:
        """Fit anomaly detection models on reference data."""
        num_data = self.reference_data[self.numerical_features]
        
        # Isolation Forest
        self.iforest = IsolationForest(
            contamination=0.1,  # Expected proportion of outliers
            random_state=self.random_state,
            n_jobs=-1
        )
        self.iforest.fit(num_data)
        
        # Robust covariance (Mahalanobis distance)
        self.robust_cov = EllipticEnvelope(
            contamination=0.1,
            random_state=self.random_state
        )
        self.robust_cov.fit(num_data)
    
    def detect_drift(
        self, 
        current_data: Union[pd.DataFrame, np.ndarray],
        methods: Optional[List[DetectionMethod]] = None,
        features: Optional[List[str]] = None,
        drift_type: DriftType = DriftType.COVARIATE_DRIFT
    ) -> Dict[str, DriftResult]:
        """
        Detect drift between reference and current data.
        
        Args:
            current_data: Current dataset to compare against reference
            methods: List of detection methods to use
            features: Features to analyze (default: all)
            drift_type: Type of drift to detect
            
        Returns:
            Dictionary of drift detection results by method
        """
        current_df = self._ensure_dataframe(current_data)
        
        # Use all features if none specified
        if features is None:
            features = self.reference_data.columns.tolist()
        
        # Use all methods if none specified
        if methods is None:
            methods = [
                DetectionMethod.STATISTICAL_TEST,
                DetectionMethod.ISOLATION_FOREST,
                DetectionMethod.MAHALANOBIS_DISTANCE
            ]
        
        results = {}
        
        # Detect drift using each method
        for method in methods:
            try:
                if method == DetectionMethod.STATISTICAL_TEST:
                    results[method.name] = self._detect_statistical_drift(current_df[features])
                elif method == DetectionMethod.ISOLATION_FOREST:
                    results[method.name] = self._detect_isolation_forest(current_df[features])
                elif method == DetectionMethod.MAHALANOBIS_DISTANCE:
                    results[method.name] = self._detect_mahalanobis(current_df[features])
                elif method == DetectionMethod.KOLMOGOROV_SMIRNOV:
                    results[method.name] = self._detect_ks_test(current_df[features])
                elif method == DetectionMethod.CHI_SQUARED:
                    results[method.name] = self._detect_chi_squared(current_df[features])
                else:
                    warnings.warn(f"Method {method.name} not implemented yet")
            except Exception as e:
                logger.error(f"Error in {method.name}: {str(e)}")
                results[method.name] = DriftResult(
                    has_drift=False,
                    drift_type=drift_type,
                    detection_method=method,
                    message=f"Error in detection: {str(e)}"
                )
        
        return results
    
    def _detect_statistical_drift(self, current_data: pd.DataFrame) -> DriftResult:
        """Detect drift using statistical tests."""
        results = []
        
        # Check numerical features
        num_features = [f for f in self.numerical_features if f in current_data.columns]
        if num_features:
            num_data = current_data[num_features]
            
            # T-test for mean comparison
            for col in num_data.columns:
                ref_mean = self.reference_stats['mean'][col]
                ref_std = self.reference_stats['std'][col]
                
                # Skip if reference std is 0 to avoid division by zero
                if ref_std == 0:
                    continue
                
                # Z-test (approximate for large samples)
                z_scores = (num_data[col].mean() - ref_mean) / (ref_std / np.sqrt(len(num_data)))
                p_value = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))
                
                results.append({
                    'feature': col,
                    'test': 'z_test',
                    'p_value': p_value,
                    'statistic': z_scores,
                    'has_drift': p_value < self.alpha
                })
        
        # Check categorical features
        cat_features = [f for f in self.categorical_features if f in current_data.columns]
        if cat_features:
            for col in cat_features:
                ref_dist = self.reference_stats['categorical'][col]['distribution']
                current_dist = current_data[col].value_counts(normalize=True).to_dict()
                
                # Align distributions
                all_categories = set(ref_dist.keys()).union(set(current_dist.keys()))
                ref_counts = [ref_dist.get(cat, 0) * len(current_data) for cat in all_categories]
                current_counts = [current_dist.get(cat, 0) * len(current_data) for cat in all_categories]
                
                # Chi-squared test if we have enough data
                if min(ref_counts) >= 5 and min(current_counts) >= 5:
                    chi2, p_value = stats.chisquare(current_counts, f_exp=ref_counts)
                    results.append({
                        'feature': col,
                        'test': 'chi2_test',
                        'p_value': p_value,
                        'statistic': chi2,
                        'has_drift': p_value < self.alpha
                    })
        
        # Aggregate results
        has_drift = any(r['has_drift'] for r in results)
        
        return DriftResult(
            has_drift=has_drift,
            drift_type=DriftType.COVARIATE_DRIFT,
            detection_method=DetectionMethod.STATISTICAL_TEST,
            p_value=min(r['p_value'] for r in results) if results else 1.0,
            test_statistic=max(abs(r['statistic']) for r in results) if results else 0.0,
            threshold=self.alpha,
            message=f"Detected drift in {sum(r['has_drift'] for r in results)}/{len(results)} features" if results else "No features to analyze",
            details={'feature_results': results}
        )
    
    def _detect_isolation_forest(self, current_data: pd.DataFrame) -> DriftResult:
        """Detect drift using Isolation Forest."""
        if not hasattr(self, 'iforest'):
            return DriftResult(
                has_drift=False,
                drift_type=DriftType.COVARIATE_DRIFT,
                detection_method=DetectionMethod.ISOLATION_FOREST,
                message="Isolation Forest not fitted on numerical features"
            )
        
        # Use only numerical features that exist in current data
        num_features = [f for f in self.numerical_features if f in current_data.columns]
        if not num_features:
            return DriftResult(
                has_drift=False,
                drift_type=DriftType.COVARIATE_DRIFT,
                detection_method=DetectionMethod.ISOLATION_FOREST,
                message="No numerical features to analyze"
            )
        
        # Get anomaly scores (lower means more anomalous)
        scores = -self.iforest.score_samples(current_data[num_features])
        
        # Calculate threshold (95th percentile of reference scores)
        ref_scores = -self.iforest.score_samples(self.reference_data[num_features])
        threshold = np.percentile(ref_scores, 95)
        
        # Calculate drift metrics
        anomaly_ratio = (scores > threshold).mean()
        has_drift = anomaly_ratio > 0.1  # More than 10% anomalies
        
        return DriftResult(
            has_drift=has_drift,
            drift_type=DriftType.COVARIATE_DRIFT,
            detection_method=DetectionMethod.ISOLATION_FOREST,
            test_statistic=float(anomaly_ratio),
            threshold=0.1,
            message=f"Anomaly ratio: {anomaly_ratio:.2f}",
            details={
                'anomaly_ratio': float(anomaly_ratio),
                'threshold': float(threshold),
                'n_anomalies': int((scores > threshold).sum())
            }
        )
    
    def _detect_mahalanobis(self, current_data: pd.DataFrame) -> DriftResult:
        """Detect drift using Mahalanobis distance."""
        if not hasattr(self, 'robust_cov'):
            return DriftResult(
                has_drift=False,
                drift_type=DriftType.COVARIATE_DRIFT,
                detection_method=DetectionMethod.MAHALANOBIS_DISTANCE,
                message="Robust covariance not fitted on numerical features"
            )
        
        # Use only numerical features that exist in current data
        num_features = [f for f in self.numerical_features if f in current_data.columns]
        if not num_features:
            return DriftResult(
                has_drift=False,
                drift_type=DriftType.COVARIATE_DRIFT,
                detection_method=DetectionMethod.MAHALANOBIS_DISTANCE,
                message="No numerical features to analyze"
            )
        
        # Get Mahalanobis distances
        distances = self.robust_cov.mahalanobis(current_data[num_features])
        
        # Calculate threshold (95th percentile of reference distances)
        ref_distances = self.robust_cov.mahalanobis(self.reference_data[num_features])
        threshold = np.percentile(ref_distances, 95)
        
        # Calculate drift metrics
        outlier_ratio = (distances > threshold).mean()
        has_drift = outlier_ratio > 0.1  # More than 10% outliers
        
        return DriftResult(
            has_drift=has_drift,
            drift_type=DriftType.COVARIATE_DRIFT,
            detection_method=DetectionMethod.MAHALANOBIS_DISTANCE,
            test_statistic=float(outlier_ratio),
            threshold=0.1,
            message=f"Outlier ratio: {outlier_ratio:.2f}",
            details={
                'outlier_ratio': float(outlier_ratio),
                'threshold': float(threshold),
                'n_outliers': int((distances > threshold).sum())
            }
        )
    
    def _detect_ks_test(self, current_data: pd.DataFrame) -> DriftResult:
        """Detect drift using Kolmogorov-Smirnov test."""
        results = []
        
        # Check numerical features
        num_features = [f for f in self.numerical_features if f in current_data.columns]
        if num_features:
            for col in num_features:
                # KS test compares empirical CDFs
                ks_stat, p_value = stats.ks_2samp(
                    self.reference_data[col].dropna(),
                    current_data[col].dropna()
                )
                
                results.append({
                    'feature': col,
                    'test': 'ks_test',
                    'p_value': p_value,
                    'statistic': ks_stat,
                    'has_drift': p_value < self.alpha
                })
        
        # Aggregate results
        has_drift = any(r['has_drift'] for r in results)
        
        return DriftResult(
            has_drift=has_drift,
            drift_type=DriftType.COVARIATE_DRIFT,
            detection_method=DetectionMethod.KOLMOGOROV_SMIRNOV,
            p_value=min(r['p_value'] for r in results) if results else 1.0,
            test_statistic=max(r['statistic'] for r in results) if results else 0.0,
            threshold=self.alpha,
            message=f"Detected drift in {sum(r['has_drift'] for r in results)}/{len(results)} features" if results else "No features to analyze",
            details={'feature_results': results}
        )
    
    def _detect_chi_squared(self, current_data: pd.DataFrame) -> DriftResult:
        """Detect drift using Chi-squared test for categorical features."""
        results = []
        
        # Check categorical features
        cat_features = [f for f in self.categorical_features if f in current_data.columns]
        if cat_features:
            for col in cat_features:
                # Get value counts for reference and current
                ref_counts = self.reference_data[col].value_counts()
                current_counts = current_data[col].value_counts()
                
                # Align indices
                all_categories = ref_counts.index.union(current_counts.index)
                ref_counts = ref_counts.reindex(all_categories, fill_value=0)
                current_counts = current_counts.reindex(all_categories, fill_value=0)
                
                # Skip if not enough data
                if (ref_counts < 5).any() or (current_counts < 5).any():
                    continue
                
                # Chi-squared test
                chi2, p_value = stats.chisquare(
                    f_obs=current_counts,
                    f_exp=ref_counts * len(current_data) / len(self.reference_data)
                )
                
                results.append({
                    'feature': col,
                    'test': 'chi2_test',
                    'p_value': p_value,
                    'statistic': chi2,
                    'has_drift': p_value < self.alpha
                })
        
        # Aggregate results
        has_drift = any(r['has_drift'] for r in results)
        
        return DriftResult(
            has_drift=has_drift,
            drift_type=DriftType.COVARIATE_DRIFT,
            detection_method=DetectionMethod.CHI_SQUARED,
            p_value=min(r['p_value'] for r in results) if results else 1.0,
            test_statistic=max(r['statistic'] for r in results) if results else 0.0,
            threshold=self.alpha,
            message=f"Detected drift in {sum(r['has_drift'] for r in results)}/{len(results)} features" if results else "No categorical features to analyze",
            details={'feature_results': results}
        )
    
    def detect_concept_drift(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        model: Any = None,
        test_size: float = 0.3,
        random_state: Optional[int] = None
    ) -> DriftResult:
        """
        Detect concept drift by comparing model performance between reference and current data.
        
        Args:
            X: Feature matrix
            y: Target vector
            model: Model to evaluate (if None, uses accuracy as metric)
            test_size: Proportion of data to use for testing
            random_state: Random seed
            
        Returns:
            DriftResult with concept drift information
        """
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values
        
        # Split reference and current data
        X_ref, X_curr, y_ref, y_curr = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Use a simple model if none provided
        if model is None:
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=50, random_state=random_state)
        
        # Train on reference data
        model.fit(X_ref, y_ref)
        
        # Evaluate on both reference and current data
        ref_pred = model.predict(X_ref)
        curr_pred = model.predict(X_curr)
        
        ref_acc = accuracy_score(y_ref, ref_pred)
        curr_acc = accuracy_score(y_curr, curr_pred)
        
        # Calculate performance drop
        perf_drop = ref_acc - curr_acc
        has_drift = perf_drop > 0.1  # More than 10% drop in accuracy
        
        return DriftResult(
            has_drift=has_drift,
            drift_type=DriftType.CONCEPT_DRIFT,
            detection_method=DetectionMethod.STATISTICAL_TEST,
            test_statistic=float(perf_drop),
            threshold=0.1,
            message=f"Performance drop: {perf_drop:.2%} (ref: {ref_acc:.2%}, curr: {curr_acc:.2%})",
            details={
                'reference_accuracy': float(ref_acc),
                'current_accuracy': float(curr_acc),
                'performance_drop': float(perf_drop)
            }
        )
    
    def monitor_drift(
        self,
        data_stream: Callable[[], pd.DataFrame],
        window_size: int = 1000,
        interval: int = 60,
        max_iter: Optional[int] = None
    ) -> None:
        """
        Monitor data drift over time.
        
        Args:
            data_stream: Function that returns the next batch of data
            window_size: Number of samples per batch
            interval: Time between checks (seconds)
            max_iter: Maximum number of iterations (None for infinite)
        """
        import time
        from collections import deque
        
        # Store drift history
        drift_history = deque(maxlen=100)
        iteration = 0
        
        while max_iter is None or iteration < max_iter:
            try:
                # Get next batch of data
                current_batch = data_stream()
                
                if len(current_batch) < window_size:
                    logger.warning(f"Batch size ({len(current_batch)}) smaller than window size ({window_size})")
                    if len(current_batch) == 0:
                        time.sleep(interval)
                        continue
                else:
                    current_batch = current_batch.sample(window_size, random_state=self.random_state)
                
                # Detect drift
                drift_results = self.detect_drift(current_batch)
                
                # Log results
                for method, result in drift_results.items():
                    if result.has_drift:
                        logger.warning(f"Drift detected by {method}: {result.message}")
                    else:
                        logger.info(f"No drift detected by {method}")
                
                # Store results
                drift_history.append({
                    'timestamp': datetime.utcnow(),
                    'results': {k: v.to_dict() for k, v in drift_results.items()}
                })
                
                iteration += 1
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in drift monitoring: {str(e)}")
                time.sleep(interval)  # Prevent tight loop on error
