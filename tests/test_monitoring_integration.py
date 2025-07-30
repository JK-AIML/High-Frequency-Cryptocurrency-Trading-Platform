"""
Integration tests for the monitoring system.

These tests verify that all components of the monitoring system work together correctly.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import asyncio

from tick_analysis.monitoring import (
    UnifiedDriftDetector,
    DriftMonitor,
    AlertRule,
    AlertSeverity,
    DriftType,
    DetectionMethod,
    DriftResult
)

# Fixtures
@pytest.fixture
def reference_data():
    """Generate reference data for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.normal(10, 2, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    })

@pytest.fixture
def current_data():
    """Generate current data with some drift for testing."""
    np.random.seed(43)  # Different seed for different data
    return pd.DataFrame({
        'feature1': np.random.normal(0.5, 1.2, 1000),  # Drifted
        'feature2': np.random.normal(10, 2, 1000),      # Not drifted
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    })

# Test cases
class TestUnifiedDriftDetector:
    """Test the UnifiedDriftDetector class."""
    
    def test_detect_drift(self, reference_data, current_data):
        """Test basic drift detection."""
        detector = UnifiedDriftDetector(
            reference_data=reference_data,
            categorical_features=['category']
        )
        
        results = detector.detect_drift(
            current_data=current_data,
            method=DetectionMethod.KOLMOGOROV_SMIRNOV
        )
        
        # Should detect drift in feature1 but not in feature2
        features = {r.feature: r for r in results}
        assert 'feature1' in features
        assert 'feature2' in features
        assert features['feature1'].is_drifted is True
        assert features['feature2'].is_drifted is False
    
    def test_multivariate_drift(self, reference_data, current_data):
        """Test multivariate drift detection."""
        detector = UnifiedDriftDetector(
            reference_data=reference_data,
            categorical_features=['category']
        )
        
        result = detector.detect_multivariate_drift(
            current_data=current_data,
            method=DetectionMethod.MAHALANOBIS
        )
        
        assert isinstance(result, DriftResult)
        assert result.drift_type == DriftType.MULTIVARIATE

class TestDriftMonitor:
    """Test the DriftMonitor class."""
    
    @pytest.mark.asyncio
    async def test_monitor_drift(self, reference_data):
        """Test monitoring for drift."""
        monitor = DriftMonitor(
            reference_data=reference_data,
            window_size=100,
            min_samples=50,
            categorical_features=['category']
        )
        
        # Mock alert handler
        mock_handler = Mock()
        monitor.add_alert_handler(mock_handler)
        
        # Generate data with drift
        for i in range(150):
            data = {
                'feature1': np.random.normal(0.5 * (i // 50), 1.0),  # Gradually increasing drift
                'feature2': np.random.normal(10, 2),
                'category': np.random.choice(['A', 'B', 'C'])
            }
            monitor.update(data)
        
        # Give some time for async processing
        await asyncio.sleep(0.1)
        
        # Should have triggered alerts for feature1
        assert mock_handler.called
        
        # Check the alert content
        alert = mock_handler.call_args[0][0]
        assert isinstance(alert, DriftResult)
        assert 'feature1' in alert.feature or 'multivariate' in str(alert.drift_type).lower()

class TestAlertRules:
    """Test alert rule functionality."""
    
    def test_alert_rule_triggering(self):
        """Test that alert rules trigger correctly."""
        # Create a test alert rule
        rule = AlertRule(
            name="test_rule",
            condition=lambda r: r.p_value is not None and r.p_value < 0.05,
            severity=AlertSeverity.WARNING,
            message_template="Test alert: {feature} has p={p_value:.4f}"
        )
        
        # Create a test drift result
        result = DriftResult(
            drift_type=DriftType.COVARIATE,
            detection_method=DetectionMethod.KOLMOGOROV_SMIRNOV,
            statistic=0.5,
            p_value=0.01,  # Should trigger the rule
            is_drifted=True,
            feature="test_feature"
        )
        
        # Should trigger the rule
        assert rule.should_trigger(result) is True
        alert = rule.create_alert(result)
        assert alert.severity == AlertSeverity.WARNING
        assert "test_feature" in alert.message
        
        # Test cooldown
        assert rule.should_trigger(result) is False

# Integration test with streaming data
class TestStreamingIntegration:
    """Test integration with streaming frameworks."""
    
    @pytest.mark.asyncio
    async def test_with_mock_stream(self, reference_data):
        """Test with a mock streaming source."""
        monitor = DriftMonitor(
            reference_data=reference_data,
            window_size=50,
            min_samples=25,
            categorical_features=['category']
        )
        
        # Mock alert handler
        alerts = []
        monitor.add_alert_handler(lambda a: alerts.append(a))
        
        # Simulate a stream of data with drift
        async def mock_stream():
            for i in range(200):
                data = {
                    'feature1': np.random.normal(0.1 * (i // 50), 1.0),  # Gradual drift
                    'feature2': np.random.normal(10, 2),
                    'category': np.random.choice(['A', 'B', 'C'])
                }
                yield data
        
        # Process the stream
        batch = []
        async for data in mock_stream():
            batch.append(data)
            if len(batch) >= 10:  # Process in small batches
                for item in batch:
                    monitor.update(item)
                batch = []
        
        # Process any remaining items
        for item in batch:
            monitor.update(item)
        
        # Should have detected some drift
        assert len(alerts) > 0
        assert any('feature1' in str(alert) for alert in alerts)
