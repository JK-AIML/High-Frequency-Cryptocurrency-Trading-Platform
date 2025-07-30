"""
Unified Drift Detection Interface

This module provides a consolidated interface for detecting various types of data drift
in time series and batch data. It combines the best features from multiple implementations
and provides a clean, consistent API.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
import warnings
from scipy import stats
from scipy.spatial.distance import jensenshannon, mahalanobis
from scipy.stats import wasserstein_distance, ks_2samp, chi2_contingency, entropy
from sklearn.ensemble import IsolationForest
from sklearn.covariance import MinCovDet
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class DriftType(Enum):
    """Types of data drift that can be detected."""
    COVARIATE = "covariate"       # Drift in feature distributions
    CONCEPT = "concept"           # Drift in target concept
    LABEL = "label"               # Drift in label distribution
    PRIOR_PROBABILITY = "prior"   # Drift in class priors
    MULTIVARIATE = "multivariate" # Multivariate feature drift
    
class DetectionMethod(Enum):
    """Available drift detection methods."""
    KOLMOGOROV_SMIRNOV = "ks_test"
    JENSEN_SHANNON = "js_divergence"
    WASSERSTEIN = "wasserstein"
    CHI_SQUARED = "chi_squared"
    PSI = "psi"
    KL_DIVERGENCE = "kl_divergence"
    MAHALANOBIS = "mahalanobis"
    ISOLATION_FOREST = "isolation_forest"
    MMD = "maximum_mean_discrepancy"

@dataclass
class DriftResult:
    """Container for drift detection results."""
    drift_type: DriftType
    detection_method: DetectionMethod
    statistic: float
    p_value: Optional[float] = None
    threshold: Optional[float] = None
    is_drifted: bool = False
    feature: Optional[str] = None
    confidence: float = 0.95
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'drift_type': self.drift_type.value,
            'detection_method': self.detection_method.value,
            'statistic': self.statistic,
            'p_value': self.p_value,
            'threshold': self.threshold,
            'is_drifted': self.is_drifted,
            'feature': self.feature,
            'confidence': self.confidence,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp.isoformat()
        }
    
    def to_json(self) -> str:
        """Convert result to JSON string."""
        return json.dumps(self.to_dict(), default=str)

class UnifiedDriftDetector:
    """
    Unified interface for detecting various types of data drift.
    
    This class provides methods to detect different types of data drift using
    statistical tests and distance metrics, with support for both univariate
    and multivariate drift detection.
    """
    
    def __init__(
        self,
        reference_data: Union[pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None,
        categorical_features: Optional[List[Union[str, int]]] = None,
        numerical_features: Optional[List[Union[str, int]]] = None,
        alpha: float = 0.05,
        random_state: int = 42,
        n_jobs: int = -1
    ):
        """
        Initialize the drift detector.
        
        Args:
            reference_data: Reference dataset (baseline)
            feature_names: Optional list of feature names
            categorical_features: List of categorical feature names/indices
            numerical_features: List of numerical feature names/indices
            alpha: Significance level for statistical tests
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs to run (-1 uses all cores)
        """
        self.reference_data = self._ensure_dataframe(reference_data, feature_names)
        self.alpha = alpha
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        # Set feature types
        self.categorical_features = categorical_features or []
        self.numerical_features = numerical_features or [
            col for col in self.reference_data.columns 
            if col not in self.categorical_features
        ]
        
        # Initialize models and scalers
        self.scaler = StandardScaler()
        self._fit_scaler()
        
    def _ensure_dataframe(
        self, 
        data: Union[pd.DataFrame, np.ndarray], 
        feature_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Ensure input is a pandas DataFrame with proper column names."""
        if isinstance(data, np.ndarray):
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(data.shape[1])]
            return pd.DataFrame(data, columns=feature_names)
        return data.copy()
    
    def _fit_scaler(self) -> None:
        """Fit the standard scaler on reference data."""
        if len(self.numerical_features) > 0:
            self.scaler.fit(self.reference_data[self.numerical_features])
    
    def detect_drift(
        self,
        current_data: Union[pd.DataFrame, np.ndarray],
        features: Optional[List[str]] = None,
        method: Union[str, DetectionMethod] = "auto",
        drift_type: Union[str, DriftType] = DriftType.COVARIATE,
        **kwargs
    ) -> List[DriftResult]:
        """
        Detect drift between reference and current data.
        
        Args:
            current_data: Current data to compare against reference
            features: List of features to analyze (None for all)
            method: Detection method to use ('auto' to choose automatically)
            drift_type: Type of drift to detect
            **kwargs: Additional method-specific arguments
            
        Returns:
            List of DriftResult objects
        """
        current_df = self._ensure_dataframe(
            current_data, 
            feature_names=features or list(self.reference_data.columns)
        )
        
        # Select features to analyze
        features = features or list(self.reference_data.columns)
        
        # Convert string enums
        if isinstance(drift_type, str):
            drift_type = DriftType(drift_type.lower())
        if isinstance(method, str):
            method = DetectionMethod(method.lower()) if method != "auto" else method
        
        # Choose appropriate detection method
        if method == "auto":
            if drift_type == DriftType.COVARIATE:
                if len(features) == 1:
                    method = DetectionMethod.KOLMOGOROV_SMIRNOV
                else:
                    method = DetectionMethod.MAHALANOBIS
            elif drift_type == DriftType.CONCEPT:
                method = DetectionMethod.JENSEN_SHANNON
            elif drift_type == DriftType.LABEL:
                method = DetectionMethod.CHI_SQUARED
            else:
                method = DetectionMethod.MAHALANOBIS
        
        # Dispatch to appropriate detection method
        if drift_type == DriftType.MULTIVARIATE:
            return self._detect_multivariate_drift(current_df, method, **kwargs)
        else:
            return self._detect_univariate_drift(current_df, features, method, drift_type, **kwargs)
    
    def _detect_univariate_drift(
        self,
        current_df: pd.DataFrame,
        features: List[str],
        method: DetectionMethod,
        drift_type: DriftType,
        **kwargs
    ) -> List[DriftResult]:
        """Detect drift for each feature individually."""
        results = []
        
        for feature in tqdm(features, desc="Detecting drift"):
            ref_data = self.reference_data[feature].dropna().values
            curr_data = current_df[feature].dropna().values
            
            if len(ref_data) == 0 or len(curr_data) == 0:
                logger.warning(f"Skipping {feature}: No valid data")
                continue
                
            if method == DetectionMethod.KOLMOGOROV_SMIRNOV:
                result = self._ks_test(ref_data, curr_data, feature, drift_type)
            elif method == DetectionMethod.JENSEN_SHANNON:
                result = self._jensen_shannon_divergence(ref_data, curr_data, feature, drift_type)
            elif method == DetectionMethod.CHI_SQUARED:
                result = self._chi_squared_test(ref_data, curr_data, feature, drift_type)
            else:
                raise ValueError(f"Unsupported method for univariate drift: {method}")
            
            results.append(result)
            
        return results
    
    def _detect_multivariate_drift(
        self,
        current_df: pd.DataFrame,
        method: DetectionMethod,
        **kwargs
    ) -> List[DriftResult]:
        """Detect drift across multiple features simultaneously."""
        if method == DetectionMethod.MAHALANOBIS:
            return [self._mahalanobis_distance(current_df, **kwargs)]
        elif method == DetectionMethod.ISOLATION_FOREST:
            return [self._isolation_forest(current_df, **kwargs)]
        else:
            raise ValueError(f"Unsupported method for multivariate drift: {method}")
    
    def _ks_test(
        self,
        ref_data: np.ndarray,
        curr_data: np.ndarray,
        feature: str,
        drift_type: DriftType
    ) -> DriftResult:
        """Kolmogorov-Smirnov test for distribution similarity."""
        stat, p_value = ks_2samp(ref_data, curr_data)
        return DriftResult(
            drift_type=drift_type,
            detection_method=DetectionMethod.KOLMOGOROV_SMIRNOV,
            statistic=stat,
            p_value=p_value,
            is_drifted=p_value < self.alpha,
            feature=feature,
            message=f"KS test {'detected' if p_value < self.alpha else 'no'} drift"
        )
    
    def _jensen_shannon_divergence(
        self,
        ref_data: np.ndarray,
        curr_data: np.ndarray,
        feature: str,
        drift_type: DriftType,
        bins: int = 10
    ) -> DriftResult:
        """Jensen-Shannon divergence between distributions."""
        # Create histograms with the same bins
        min_val = min(np.min(ref_data), np.min(curr_data))
        max_val = max(np.max(ref_data), np.max(curr_data))
        bins = np.linspace(min_val, max_val, bins + 1)
        
        hist_ref = np.histogram(ref_data, bins=bins, density=True)[0]
        hist_curr = np.histogram(curr_data, bins=bins, density=True)[0]
        
        # Add small constant to avoid division by zero
        hist_ref = hist_ref + 1e-10
        hist_curr = hist_curr + 1e-10
        
        # Normalize
        hist_ref = hist_ref / np.sum(hist_ref)
        hist_curr = hist_curr / np.sum(hist_curr)
        
        # Calculate JS divergence
        m = 0.5 * (hist_ref + hist_curr)
        js_div = 0.5 * (entropy(hist_ref, m) + entropy(hist_curr, m))
        
        return DriftResult(
            drift_type=drift_type,
            detection_method=DetectionMethod.JENSEN_SHANNON,
            statistic=js_div,
            threshold=0.1,  # Common threshold for JS divergence
            is_drifted=js_div > 0.1,
            feature=feature,
            message=f"JS divergence {'exceeded' if js_div > 0.1 else 'below'} threshold"
        )
    
    def _chi_squared_test(
        self,
        ref_data: np.ndarray,
        curr_data: np.ndarray,
        feature: str,
        drift_type: DriftType,
        bins: int = 10
    ) -> DriftResult:
        """Chi-squared test for categorical or binned numerical data."""
        # Create histograms with the same bins
        min_val = min(np.min(ref_data), np.min(curr_data))
        max_val = max(np.max(ref_data), np.max(curr_data))
        bins = np.linspace(min_val, max_val, bins + 1)
        
        hist_ref = np.histogram(ref_data, bins=bins)[0]
        hist_curr = np.histogram(curr_data, bins=bins)[0]
        
        # Perform chi-squared test
        chi2, p_value, dof, _ = chi2_contingency([hist_ref, hist_curr])
        
        return DriftResult(
            drift_type=drift_type,
            detection_method=DetectionMethod.CHI_SQUARED,
            statistic=chi2,
            p_value=p_value,
            is_drifted=p_value < self.alpha,
            feature=feature,
            message=f"Chi-squared test {'detected' if p_value < self.alpha else 'no'} drift"
        )
    
    def _mahalanobis_distance(
        self,
        current_df: pd.DataFrame,
        **kwargs
    ) -> DriftResult:
        """Calculate Mahalanobis distance between reference and current data."""
        # Use only numerical features
        features = [f for f in self.numerical_features if f in current_df.columns]
        
        if not features:
            raise ValueError("No numerical features available for Mahalanobis distance")
        
        # Scale the data
        ref_scaled = self.scaler.transform(self.reference_data[features])
        curr_scaled = self.scaler.transform(current_df[features])
        
        # Calculate covariance matrix and its inverse
        cov = np.cov(ref_scaled, rowvar=False)
        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            # Handle singular matrix by adding small diagonal matrix
            cov_inv = np.linalg.pinv(cov + np.eye(cov.shape[0]) * 1e-6)
        
        # Calculate means
        ref_mean = np.mean(ref_scaled, axis=0)
        curr_mean = np.mean(curr_scaled, axis=0)
        
        # Calculate Mahalanobis distance
        mean_diff = curr_mean - ref_mean
        mahal_dist = np.sqrt(mean_diff.T @ cov_inv @ mean_diff)
        
        # Simple threshold (can be made more sophisticated)
        threshold = 3.0  # Common threshold for Mahalanobis distance
        
        return DriftResult(
            drift_type=DriftType.MULTIVARIATE,
            detection_method=DetectionMethod.MAHALANOBIS,
            statistic=mahal_dist,
            threshold=threshold,
            is_drifted=mahal_dist > threshold,
            message=f"Mahalanobis distance {'exceeded' if mahal_dist > threshold else 'below'} threshold",
            details={
                'features': features,
                'reference_mean': ref_mean.tolist(),
                'current_mean': curr_mean.tolist()
            }
        )
    
    def _isolation_forest(
        self,
        current_df: pd.DataFrame,
        contamination: float = 0.1,
        **kwargs
    ) -> DriftResult:
        """Use Isolation Forest to detect anomalous samples in current data."""
        # Use only numerical features
        features = [f for f in self.numerical_features if f in current_df.columns]
        
        if not features:
            raise ValueError("No numerical features available for Isolation Forest")
        
        # Scale the data
        ref_scaled = self.scaler.transform(self.reference_data[features])
        curr_scaled = self.scaler.transform(current_df[features])
        
        # Train Isolation Forest on reference data
        clf = IsolationForest(
            contamination=contamination,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        clf.fit(ref_scaled)
        
        # Predict anomalies in current data
        preds = clf.predict(curr_scaled)
        anomaly_ratio = np.mean(preds == -1)
        
        return DriftResult(
            drift_type=DriftType.MULTIVARIATE,
            detection_method=DetectionMethod.ISOLATION_FOREST,
            statistic=anomaly_ratio,
            threshold=contamination * 2,  # Allow twice the expected contamination
            is_drifted=anomaly_ratio > (contamination * 2),
            message=f"Anomaly ratio: {anomaly_ratio:.3f} (threshold: {contamination * 2:.3f})",
            details={
                'features': features,
                'contamination': contamination,
                'anomaly_ratio': float(anomaly_ratio)
            }
        )

# Example usage
if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Generate example data
    np.random.seed(42)
    ref_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.normal(5, 2, 1000),
        'categorical': np.random.choice(['A', 'B', 'C'], size=1000)
    })
    
    # Current data with some drift
    curr_data = pd.DataFrame({
        'feature1': np.random.normal(0.5, 1.2, 1000),  # Some drift
        'feature2': np.random.normal(5, 2, 1000),       # No drift
        'categorical': np.random.choice(['A', 'B', 'C', 'D'], size=1000)  # New category
    })
    
    # Initialize detector
    detector = UnifiedDriftDetector(
        reference_data=ref_data,
        categorical_features=['categorical']
    )
    
    # Detect univariate drift
    print("\nDetecting univariate drift:")
    results = detector.detect_drift(
        current_data=curr_data,
        drift_type=DriftType.COVARIATE,
        method="auto"
    )
    
    for result in results:
        print(f"{result.feature}: {result.message}")
    
    # Detect multivariate drift
    print("\nDetecting multivariate drift:")
    result = detector.detect_drift(
        current_data=curr_data,
        drift_type=DriftType.MULTIVARIATE,
        method="mahalanobis"
    )[0]
    
    print(f"Multivariate drift detected: {result.is_drifted}")
    print(f"Mahalanobis distance: {result.statistic:.3f} (threshold: {result.threshold})")
