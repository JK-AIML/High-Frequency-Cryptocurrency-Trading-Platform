"""
Data Drift Detection Module

This module provides functionality for detecting drift in data distributions
and feature values over time, which is crucial for maintaining model performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress scipy warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)

class DriftType(Enum):
    """Types of data drift that can be detected."""
    COVARIATE = "covariate"  # Input feature distribution change
    LABEL = "label"          # Target distribution change
    CONCEPT = "concept"      # Relationship between features and target changes
    ANOMALY = "anomaly"      # Unusual data points

class StatisticalTest(Enum):
    """Statistical tests for drift detection."""
    KS_TEST = "ks"           # Kolmogorov-Smirnov test
    CHI_SQUARED = "chi2"      # Chi-squared test
    ANDERSON = "anderson"     # Anderson-Darling test
    T_TEST = "ttest"          # Student's t-test
    MANNWHITNEY = "mannwhitney"  # Mann-Whitney U test
    WELCH = "welch"           # Welch's t-test
    KL_DIVERGENCE = "kldiv"   # Kullback-Leibler divergence
    JSD = "jsd"               # Jensen-Shannon divergence
    WASSERSTEIN = "wasserstein"  # Wasserstein distance
    ENERGY_DIST = "energy"    # Energy distance

@dataclass
class DriftResult:
    """Result of a drift detection test."""
    detected: bool
    score: float
    p_value: Optional[float] = None
    threshold: Optional[float] = None
    test_type: Optional[StatisticalTest] = None
    feature: Optional[str] = None
    drift_type: DriftType = DriftType.COVARIATE
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            'detected': self.detected,
            'score': self.score,
            'p_value': self.p_value,
            'threshold': self.threshold,
            'test_type': self.test_type.value if self.test_type else None,
            'feature': self.feature,
            'drift_type': self.drift_type.value,
            'metadata': self.metadata
        }
        return result

class DataDriftDetector:
    """
    Detects drift in data distributions between reference and current data.
    
    This class provides various statistical tests and distance metrics to detect
    changes in data distributions over time, which can indicate model degradation.
    """
    
    def __init__(self, 
                 reference_data: Union[pd.DataFrame, np.ndarray],
                 feature_names: List[str] = None,
                 scaling: str = 'standard',
                 contamination: float = 0.01):
        """
        Initialize the drift detector with reference data.
        
        Args:
            reference_data: Reference dataset (pandas DataFrame or numpy array)
            feature_names: List of feature names (required if reference_data is numpy array)
            scaling: Type of scaling to apply ('standard', 'minmax', or None)
            contamination: Expected proportion of outliers in the data (for anomaly detection)
        """
        self.reference_data = reference_data
        self.feature_names = feature_names
        self.scaling = scaling
        self.scaler = None
        self.contamination = contamination
        self.reference_stats = {}
        
        # Convert numpy array to DataFrame if needed
        if isinstance(reference_data, np.ndarray):
            if feature_names is None:
                feature_names = [f'feature_{i}' for i in range(reference_data.shape[1])]
            self.reference_data = pd.DataFrame(reference_data, columns=feature_names)
        
        # Initialize scaler if needed
        if self.scaling == 'standard':
            self.scaler = StandardScaler()
            self.scaler.fit(self.reference_data)
        elif self.scaling == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler()
            self.scaler.fit(self.reference_data)
        
        # Compute reference statistics
        self._compute_reference_stats()
    
    def _compute_reference_stats(self) -> None:
        """Compute reference statistics for drift detection."""
        if self.scaler is not None:
            scaled_data = self.scaler.transform(self.reference_data)
            self.reference_data_scaled = pd.DataFrame(
                scaled_data, 
                columns=self.reference_data.columns
            )
        else:
            self.reference_data_scaled = self.reference_data.copy()
        
        # Store basic statistics
        self.reference_stats['mean'] = self.reference_data_scaled.mean()
        self.reference_stats['std'] = self.reference_data_scaled.std()
        self.reference_stats['min'] = self.reference_data_scaled.min()
        self.reference_stats['max'] = self.reference_data_scaled.max()
        self.reference_stats['skew'] = self.reference_data_scaled.skew()
        self.reference_stats['kurtosis'] = self.reference_data_scaled.kurtosis()
    
    def detect_drift(self, 
                    current_data: Union[pd.DataFrame, np.ndarray],
                    test_type: StatisticalTest = StatisticalTest.KS_TEST,
                    alpha: float = 0.05,
                    features: List[str] = None,
                    drift_type: DriftType = DriftType.COVARIATE) -> Dict[str, DriftResult]:
        """
        Detect drift between reference and current data.
        
        Args:
            current_data: Current dataset to compare against reference
            test_type: Statistical test to use for drift detection
            alpha: Significance level for statistical tests
            features: List of features to analyze (None for all)
            drift_type: Type of drift to detect
            
        Returns:
            Dictionary mapping feature names to DriftResult objects
        """
        # Convert current data to DataFrame if needed
        if isinstance(current_data, np.ndarray):
            if features is None and hasattr(self, 'feature_names'):
                features = self.feature_names
            current_data = pd.DataFrame(current_data, columns=features)
        
        # Apply same scaling as reference data
        if self.scaler is not None:
            current_data_scaled = pd.DataFrame(
                self.scaler.transform(current_data),
                columns=current_data.columns
            )
        else:
            current_data_scaled = current_data.copy()
        
        # Select features to analyze
        if features is None:
            features = current_data.columns.tolist()
        
        results = {}
        
        for feature in features:
            ref_data = self.reference_data_scaled[feature].dropna()
            curr_data = current_data_scaled[feature].dropna()
            
            # Skip if not enough data
            if len(ref_data) < 10 or len(curr_data) < 10:
                results[feature] = DriftResult(
                    detected=False,
                    score=0,
                    p_value=1.0,
                    threshold=alpha,
                    test_type=test_type,
                    feature=feature,
                    drift_type=drift_type,
                    metadata={'warning': 'Insufficient data'}
                )
                continue
            
            # Apply selected test
            if test_type == StatisticalTest.KS_TEST:
                score, p_value = stats.ks_2samp(ref_data, curr_data)
                threshold = alpha
                detected = p_value < threshold
            
            elif test_type == StatisticalTest.T_TEST:
                t_stat, p_value = stats.ttest_ind(ref_data, curr_data, equal_var=False)
                score = abs(t_stat)
                threshold = stats.t.ppf(1 - alpha/2, min(len(ref_data), len(curr_data)) - 1)
                detected = p_value < alpha
            
            elif test_type == StatisticalTest.MANNWHITNEY:
                u_stat, p_value = stats.mannwhitneyu(ref_data, curr_data, alternative='two-sided')
                score = u_stat
                threshold = alpha
                detected = p_value < threshold
            
            elif test_type == StatisticalTest.KL_DIVERGENCE:
                from scipy.special import kl_div
                # Create histograms with same bins
                hist1, bin_edges = np.histogram(ref_data, bins='auto', density=True)
                hist2, _ = np.histogram(curr_data, bins=bin_edges, density=True)
                # Add small constant to avoid division by zero
                hist1 = np.clip(hist1, 1e-10, None)
                hist2 = np.clip(hist2, 1e-10, None)
                # Calculate KL divergence (not symmetric)
                score = np.sum(kl_div(hist1, hist2))
                # Simple threshold-based detection
                threshold = 0.1  # Arbitrary threshold
                detected = score > threshold
                p_value = None
            
            elif test_type == StatisticalTest.JSD:
                from scipy.spatial.distance import jensenshannon
                # Create histograms with same bins
                hist1, bin_edges = np.histogram(ref_data, bins='auto', density=True)
                hist2, _ = np.histogram(curr_data, bins=bin_edges, density=True)
                # Calculate Jensen-Shannon divergence
                score = jensenshannon(hist1, hist2)
                # Simple threshold-based detection
                threshold = 0.1  # Arbitrary threshold
                detected = score > threshold
                p_value = None
            
            elif test_type == StatisticalTest.WASSERSTEIN:
                from scipy.stats import wasserstein_distance
                score = wasserstein_distance(ref_data, curr_data)
                # Simple threshold-based detection
                threshold = 0.1  # Arbitrary threshold
                detected = score > threshold
                p_value = None
            
            else:
                raise ValueError(f"Unsupported test type: {test_type}")
            
            results[feature] = DriftResult(
                detected=detected,
                score=float(score),
                p_value=float(p_value) if p_value is not None else None,
                threshold=float(threshold) if threshold is not None else None,
                test_type=test_type,
                feature=feature,
                drift_type=drift_type,
                metadata={
                    'reference_size': len(ref_data),
                    'current_size': len(curr_data)
                }
            )
        
        return results
    
    def detect_feature_drift(self, 
                           current_data: Union[pd.DataFrame, np.ndarray],
                           alpha: float = 0.05,
                           features: List[str] = None) -> Dict[str, DriftResult]:
        """
        Detect drift in feature distributions.
        
        Args:
            current_data: Current dataset to compare against reference
            alpha: Significance level for statistical tests
            features: List of features to analyze (None for all)
            
        Returns:
            Dictionary mapping feature names to DriftResult objects
        """
        return self.detect_drift(
            current_data=current_data,
            test_type=StatisticalTest.KS_TEST,
            alpha=alpha,
            features=features,
            drift_type=DriftType.COVARIATE
        )
    
    def detect_concept_drift(self,
                           X_reference: Union[pd.DataFrame, np.ndarray],
                           y_reference: Union[pd.Series, np.ndarray],
                           X_current: Union[pd.DataFrame, np.ndarray],
                           y_current: Union[pd.Series, np.ndarray],
                           alpha: float = 0.05) -> Dict[str, DriftResult]:
        """
        Detect concept drift by comparing feature-target relationships.
        
        Args:
            X_reference: Reference feature matrix
            y_reference: Reference target values
            X_current: Current feature matrix
            y_current: Current target values
            alpha: Significance level for statistical tests
            
        Returns:
            Dictionary with concept drift detection results
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import accuracy_score
        
        # Combine reference and current data with labels
        X_combined = np.vstack([X_reference, X_current])
        y_domain = np.array([0] * len(X_reference) + [1] * len(X_current))
        
        # Train a classifier to distinguish between reference and current data
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        
        # Use cross-validated accuracy as drift score
        scores = cross_val_score(clf, X_combined, y_domain, cv=5, scoring='accuracy')
        avg_accuracy = np.mean(scores)
        
        # If the classifier can distinguish between the datasets, there's likely drift
        p_value = 1.0 - avg_accuracy  # Simple heuristic
        detected = p_value < alpha
        
        result = DriftResult(
            detected=detected,
            score=float(avg_accuracy),
            p_value=float(p_value),
            threshold=1.0 - alpha,
            test_type=StatisticalTest.T_TEST,
            feature='concept_drift',
            drift_type=DriftType.CONCEPT,
            metadata={
                'reference_samples': len(X_reference),
                'current_samples': len(X_current),
                'cv_scores': scores.tolist()
            }
        )
        
        return {'concept_drift': result}
    
    def detect_anomalies(self,
                       data: Union[pd.DataFrame, np.ndarray],
                       method: str = 'isolation_forest',
                       contamination: float = None) -> Dict[str, Any]:
        """
        Detect anomalous data points.
        
        Args:
            data: Data to analyze for anomalies
            method: Detection method ('isolation_forest' or 'elliptic_envelope')
            contamination: Expected proportion of outliers (overrides constructor)
            
        Returns:
            Dictionary with anomaly detection results
        """
        if contamination is None:
            contamination = self.contamination
        
        # Convert to numpy array if needed
        if isinstance(data, pd.DataFrame):
            data_array = data.values
        else:
            data_array = data
        
        # Apply scaling if configured
        if self.scaler is not None:
            data_scaled = self.scaler.transform(data_array)
        else:
            data_scaled = data_array
        
        # Initialize and fit detector
        if method == 'isolation_forest':
            detector = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100
            )
        elif method == 'elliptic_envelope':
            detector = EllipticEnvelope(
                contamination=contamination,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported anomaly detection method: {method}")
        
        # Fit and predict
        is_anomaly = detector.fit_predict(data_scaled)
        anomaly_scores = detector.score_samples(data_scaled)
        
        # Convert to boolean mask (True = anomaly)
        is_anomaly = is_anomaly == -1
        
        return {
            'is_anomaly': is_anomaly,
            'anomaly_scores': anomaly_scores,
            'num_anomalies': int(np.sum(is_anomaly)),
            'contamination': contamination,
            'method': method
        }

# Example usage
if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Generate example data
    np.random.seed(42)
    n_samples = 1000
    
    # Reference data (normal distribution)
    ref_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(5, 2, n_samples),
        'feature3': np.random.exponential(1, n_samples)
    })
    
    # Current data (with some drift)
    curr_data = pd.DataFrame({
        'feature1': np.random.normal(0.5, 1.2, n_samples),  # Shifted mean and increased variance
        'feature2': np.random.normal(5, 2, n_samples),      # Same distribution
        'feature3': np.random.normal(0, 1, n_samples)       # Completely different distribution
    })
    
    # Initialize detector with reference data
    detector = DataDriftDetector(
        reference_data=ref_data,
        scaling='standard',
        contamination=0.01
    )
    
    # Detect feature drift
    print("Detecting feature drift...")
    drift_results = detector.detect_feature_drift(curr_data, alpha=0.01)
    
    # Print results
    for feature, result in drift_results.items():
        print(f"\nFeature: {feature}")
        print(f"  Drift detected: {result.detected}")
        print(f"  Test: {result.test_type.value}")
        print(f"  Score: {result.score:.4f}")
        if result.p_value is not None:
            print(f"  p-value: {result.p_value:.4f}")
        print(f"  Threshold: {result.threshold}")
    
    # Detect anomalies in current data
    print("\nDetecting anomalies...")
    anomaly_results = detector.detect_anomalies(curr_data, method='isolation_forest')
    print(f"Found {anomaly_results['num_anomalies']} anomalies "
          f"(expected ~{int(len(curr_data) * anomaly_results['contamination'])})")
