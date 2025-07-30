"""
Tests for the UnifiedDriftDetector class.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tick_analysis.monitoring.unified_drift import (
    UnifiedDriftDetector,
    DriftType,
    DetectionMethod,
    DriftResult
)

# Test data
def generate_test_data(n_samples=1000, seed=42):
    """Generate test data with controlled drift."""
    np.random.seed(seed)
    
    # Reference data
    ref_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(5, 2, n_samples),
        'categorical': np.random.choice(['A', 'B', 'C'], size=n_samples, p=[0.6, 0.3, 0.1])
    })
    
    # Current data with different levels of drift
    no_drift = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(5, 2, n_samples),
        'categorical': np.random.choice(['A', 'B', 'C'], size=n_samples, p=[0.6, 0.3, 0.1])
    })
    
    mild_drift = pd.DataFrame({
        'feature1': np.random.normal(0.2, 1.1, n_samples),
        'feature2': np.random.normal(5.5, 2.1, n_samples),
        'categorical': np.random.choice(['A', 'B', 'C', 'D'], size=n_samples, p=[0.55, 0.25, 0.1, 0.1])
    })
    
    severe_drift = pd.DataFrame({
        'feature1': np.random.normal(1.5, 2, n_samples),
        'feature2': np.random.normal(8, 3, n_samples),
        'categorical': np.random.choice(['D', 'E', 'F'], size=n_samples, p=[0.6, 0.3, 0.1])
    })
    
    return ref_data, no_drift, mild_drift, severe_drift

class TestUnifiedDriftDetector:
    """Test cases for UnifiedDriftDetector."""
    
    @classmethod
    def setup_class(cls):
        """Set up test data."""
        cls.ref_data, cls.no_drift, cls.mild_drift, cls.severe_drift = generate_test_data()
        
        # Initialize detector
        cls.detector = UnifiedDriftDetector(
            reference_data=cls.ref_data,
            categorical_features=['categorical'],
            alpha=0.05,
            random_state=42
        )
    
    def test_init(self):
        """Test detector initialization."""
        assert self.detector is not None
        assert hasattr(self.detector, 'reference_data')
        assert hasattr(self.detector, 'alpha')
        assert hasattr(self.detector, 'random_state')
        assert len(self.detector.numerical_features) == 2
        assert 'categorical' in self.detector.categorical_features
    
    def test_ks_test_no_drift(self):
        """Test KS test with no drift."""
        results = self.detector.detect_drift(
            current_data=self.no_drift,
            features=['feature1'],
            method=DetectionMethod.KOLMOGOROV_SMIRNOV,
            drift_type=DriftType.COVARIATE
        )
        
        assert len(results) == 1
        result = results[0]
        assert isinstance(result, DriftResult)
        assert result.drift_type == DriftType.COVARIATE
        assert result.detection_method == DetectionMethod.KOLMOGOROV_SMIRNOV
        assert not result.is_drifted
    
    def test_ks_test_with_drift(self):
        """Test KS test with drift."""
        results = self.detector.detect_drift(
            current_data=self.severe_drift,
            features=['feature1'],
            method=DetectionMethod.KOLMOGOROV_SMIRNOV,
            drift_type=DriftType.COVARIATE
        )
        
        assert len(results) == 1
        result = results[0]
        assert result.is_drifted
        assert result.p_value < self.detector.alpha
    
    def test_chi_squared_test(self):
        """Test chi-squared test for categorical data."""
        results = self.detector.detect_drift(
            current_data=self.mild_drift,
            features=['categorical'],
            method=DetectionMethod.CHI_SQUARED,
            drift_type=DriftType.LABEL
        )
        
        assert len(results) == 1
        result = results[0]
        assert result.drift_type == DriftType.LABEL
        assert result.detection_method == DetectionMethod.CHI_SQUARED
        assert result.is_drifted
    
    def test_jensen_shannon_divergence(self):
        """Test Jensen-Shannon divergence."""
        results = self.detector.detect_drift(
            current_data=self.mild_drift,
            features=['feature1'],
            method=DetectionMethod.JENSEN_SHANNON,
            drift_type=DriftType.COVARIATE
        )
        
        assert len(results) == 1
        result = results[0]
        assert isinstance(result, DriftResult)
        assert 0 <= result.statistic <= 1
    
    def test_mahalanobis_distance(self):
        """Test Mahalanobis distance for multivariate drift."""
        results = self.detector.detect_drift(
            current_data=self.severe_drift,
            drift_type=DriftType.MULTIVARIATE,
            method=DetectionMethod.MAHALANOBIS
        )
        
        assert len(results) == 1
        result = results[0]
        assert result.drift_type == DriftType.MULTIVARIATE
        assert result.detection_method == DetectionMethod.MAHALANOBIS
        assert result.is_drifted
        assert result.statistic > result.threshold
    
    def test_isolation_forest(self):
        """Test Isolation Forest for anomaly detection."""
        results = self.detector.detect_drift(
            current_data=self.severe_drift,
            drift_type=DriftType.MULTIVARIATE,
            method=DetectionMethod.ISOLATION_FOREST,
            contamination=0.1
        )
        
        assert len(results) == 1
        result = results[0]
        assert result.drift_type == DriftType.MULTIVARIATE
        assert result.detection_method == DetectionMethod.ISOLATION_FOREST
        assert result.is_drifted
    
    def test_auto_detection(self):
        """Test automatic method selection."""
        # Univariate case
        results = self.detector.detect_drift(
            current_data=self.mild_drift,
            features=['feature1'],
            method='auto',
            drift_type=DriftType.COVARIATE
        )
        assert len(results) == 1
        assert results[0].detection_method == DetectionMethod.KOLMOGOROV_SMIRNOV
        
        # Multivariate case
        results = self.detector.detect_drift(
            current_data=self.mild_drift,
            method='auto',
            drift_type=DriftType.MULTIVARIATE
        )
        assert len(results) == 1
        assert results[0].detection_method == DetectionMethod.MAHALANOBIS
    
    def test_result_serialization(self):
        """Test serialization of DriftResult."""
        result = DriftResult(
            drift_type=DriftType.COVARIATE,
            detection_method=DetectionMethod.KOLMOGOROV_SMIRNOV,
            statistic=0.5,
            p_value=0.01,
            is_drifted=True,
            feature='test_feature',
            message='Test message'
        )
        
        # Test to_dict
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict['drift_type'] == 'covariate'
        assert result_dict['detection_method'] == 'ks_test'
        assert result_dict['is_drifted'] is True
        
        # Test to_json
        json_str = result.to_json()
        assert isinstance(json_str, str)
        assert 'test_feature' in json_str

if __name__ == "__main__":
    pytest.main(["-v", "test_unified_drift.py"])
