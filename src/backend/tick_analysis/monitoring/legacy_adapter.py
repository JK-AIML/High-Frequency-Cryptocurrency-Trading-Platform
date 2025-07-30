"""
Legacy Adapter for Drift Detection

This module provides compatibility with older drift detection implementations
by adapting them to use the new UnifiedDriftDetector.
"""

from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import pandas as pd
from datetime import datetime

from .unified_drift import (
    UnifiedDriftDetector,
    DriftType,
    DetectionMethod,
    DriftResult as UnifiedDriftResult
)

class LegacyDriftAdapter:
    """
    Adapter for legacy drift detection code to use the new UnifiedDriftDetector.
    
    This class provides a compatibility layer that allows existing code to work
    with the new unified drift detection system.
    """
    
    def __init__(self, reference_data: Union[pd.DataFrame, np.ndarray], **kwargs):
        """
        Initialize the adapter with reference data.
        
        Args:
            reference_data: Reference dataset for drift detection
            **kwargs: Additional arguments for UnifiedDriftDetector
        """
        self.detector = UnifiedDriftDetector(reference_data, **kwargs)
    
    def detect_drift(
        self,
        current_data: Union[pd.DataFrame, np.ndarray],
        features: Optional[List[str]] = None,
        method: str = 'ks_test',
        drift_type: str = 'covariate',
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Detect drift between reference and current data.
        
        Args:
            current_data: Current data to compare against reference
            features: List of features to analyze (None for all)
            method: Detection method to use
            drift_type: Type of drift to detect
            **kwargs: Additional method-specific arguments
            
        Returns:
            List of drift detection results as dictionaries
        """
        # Map legacy method names to DetectionMethod enum
        method_map = {
            'ks': DetectionMethod.KOLMOGOROV_SMIRNOV,
            'chi2': DetectionMethod.CHI_SQUARED,
            'jsd': DetectionMethod.JENSEN_SHANNON,
            'wasserstein': DetectionMethod.WASSERSTEIN,
            'kl_divergence': DetectionMethod.KL_DIVERGENCE,
            'isolation_forest': DetectionMethod.ISOLATION_FOREST,
            'mahalanobis': DetectionMethod.MAHALANOBIS
        }
        
        # Map legacy drift types to DriftType enum
        drift_type_map = {
            'covariate': DriftType.COVARIATE,
            'concept': DriftType.CONCEPT,
            'label': DriftType.LABEL,
            'anomaly': DriftType.ANOMALY
        }
        
        # Convert to enums
        try:
            detection_method = method_map.get(method.lower(), DetectionMethod.KOLMOGOROV_SMIRNOV)
            drift_type_enum = drift_type_map.get(drift_type.lower(), DriftType.COVARIATE)
        except (AttributeError, KeyError):
            # Default to KS test if method is not recognized
            detection_method = DetectionMethod.KOLMOGOROV_SMIRNOV
            drift_type_enum = DriftType.COVARIATE
        
        # Detect drift using the unified detector
        results = self.detector.detect_drift(
            current_data=current_data,
            features=features,
            method=detection_method,
            drift_type=drift_type_enum,
            **kwargs
        )
        
        # Convert to legacy format
        return [self._to_legacy_result(r) for r in results]
    
    def _to_legacy_result(self, result: UnifiedDriftResult) -> Dict[str, Any]:
        """Convert a UnifiedDriftResult to the legacy format."""
        return {
            'detected': result.is_drifted,
            'score': result.statistic,
            'p_value': result.p_value,
            'threshold': result.threshold,
            'test_type': result.detection_method.value,
            'feature': result.feature,
            'drift_type': result.drift_type.value,
            'metadata': {
                'message': result.message,
                'confidence': result.confidence,
                'timestamp': result.timestamp.isoformat()
            }
        }
    
    def get_reference_stats(self) -> Dict[str, Any]:
        """Get statistics about the reference data."""
        # This is a simplified version - in a real implementation, you would
        # compute actual statistics from the reference data
        return {
            'n_samples': len(self.detector.reference_data),
            'n_features': len(self.detector.reference_data.columns),
            'feature_names': list(self.detector.reference_data.columns)
        }
    
    def set_reference_data(self, reference_data: Union[pd.DataFrame, np.ndarray]) -> None:
        """Update the reference data."""
        self.detector = UnifiedDriftDetector(
            reference_data=reference_data,
            **self.detector.__dict__
        )

# Backward compatibility aliases
DataDriftDetector = LegacyDriftAdapter
DriftDetector = LegacyDriftAdapter
