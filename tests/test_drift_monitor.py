"""
Tests for the drift monitoring system.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import asyncio

from tick_analysis.monitoring.drift_monitor import (
    DriftMonitor,
    AlertRule,
    AlertSeverity,
    DriftAlert
)
from tick_analysis.monitoring.unified_drift import DriftResult, DriftType, DetectionMethod

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
def drift_monitor(reference_data):
    """Create a DriftMonitor instance for testing."""
    return DriftMonitor(
        reference_data=reference_data,
        categorical_features=['category'],
        window_size=100,
        min_samples=50
    )

# Test cases
class TestAlertRule:
    """Test cases for AlertRule class."""
    
    def test_should_trigger(self):
        """Test alert rule triggering logic."""
        # Create a test drift result
        result = DriftResult(
            drift_type=DriftType.COVARIATE,
            detection_method=DetectionMethod.KOLMOGOROV_SMIRNOV,
            statistic=0.5,
            p_value=0.01,
            is_drifted=True,
            feature='test_feature'
        )
        
        # Rule that triggers on p_value < 0.05
        rule = AlertRule(
            name='test_rule',
            condition=lambda r: r.p_value is not None and r.p_value < 0.05,
            severity=AlertSeverity.WARNING
        )
        
        assert rule.should_trigger(result) is True
        
        # Should respect cooldown
        assert rule.should_trigger(result) is False
    
    def test_create_alert(self):
        """Test alert creation."""
        result = DriftResult(
            drift_type=DriftType.COVARIATE,
            detection_method=DetectionMethod.KOLMOGOROV_SMIRNOV,
            statistic=0.7,
            p_value=0.001,
            is_drifted=True,
            feature='test_feature',
            message="Drift detected"
        )
        
        rule = AlertRule(
            name='test_rule',
            condition=lambda r: True,
            severity=AlertSeverity.CRITICAL,
            message_template="Alert: {feature} has drifted with p={p_value:.4f}"
        )
        
        alert = rule.create_alert(result, {'extra': 'data'})
        
        assert alert.severity == AlertSeverity.CRITICAL
        assert 'test_feature' in alert.message
        assert alert.drift_result == result
        assert alert.metadata['extra'] == 'data'

class TestDriftMonitor:
    """Test cases for DriftMonitor class."""
    
    @pytest.mark.asyncio
    async def test_alert_handler(self, drift_monitor):
        """Test alert handler registration and calling."""
        mock_handler = AsyncMock()
        drift_monitor.add_alert_handler(mock_handler)
        
        # Create a test alert
        alert = DriftAlert(
            timestamp=datetime.utcnow(),
            severity=AlertSeverity.WARNING,
            message="Test alert",
            drift_result=DriftResult(
                drift_type=DriftType.COVARIATE,
                detection_method=DetectionMethod.KOLMOGOROV_SMIRNOV,
                statistic=0.5,
                p_value=0.01,
                is_drifted=True,
                feature='test_feature'
            )
        )
        
        # Trigger the alert handler
        await drift_monitor._handle_alert(alert)
        
        # Check that the handler was called with the alert
        mock_handler.assert_awaited_once_with(alert)
    
    def test_update_with_drift(self, drift_monitor):
        """Test updating monitor with drifting data."""
        # Add a simple alert handler that just records alerts
        alerts = []
        drift_monitor.add_alert_handler(lambda a: alerts.append(a))
        
        # Generate data with drift (different distribution than reference)
        for _ in range(200):
            data = {
                'feature1': np.random.normal(5, 1),  # Different mean
                'feature2': np.random.normal(10, 2),
                'category': np.random.choice(['A', 'B', 'C'])
            }
            drift_monitor.update(data)
        
        # We should have received some alerts
        assert len(alerts) > 0
        assert any('feature1' in alert.message for alert in alerts)
    
    @pytest.mark.asyncio
    async def test_monitor_stream(self, drift_monitor):
        """Test monitoring a stream of data."""
        # Create a simple async data stream
        async def mock_data_stream():
            for i in range(150):  # More than batch size
                yield {
                    'feature1': np.random.normal(0.1 * (i // 50), 1.0),  # Gradually drifting
                    'feature2': np.random.normal(10, 2),
                    'category': np.random.choice(['A', 'B', 'C'])
                }
        
        # Add a mock alert handler
        mock_handler = AsyncMock()
        drift_monitor.add_alert_handler(mock_handler)
        
        # Run the monitor
        await drift_monitor.monitor_stream(mock_data_stream(), batch_size=50)
        
        # Check that the handler was called (we expect some alerts due to drift)
        assert mock_handler.await_count > 0

# Integration test with real drift detection
class TestIntegration:
    """Integration tests with real drift detection."""
    
    def test_end_to_end_detection(self, reference_data):
        """Test end-to-end drift detection."""
        # Create monitor with small window for testing
        monitor = DriftMonitor(
            reference_data=reference_data,
            categorical_features=['category'],
            window_size=50,
            min_samples=20
        )
        
        # Add a simple alert handler
        alerts = []
        monitor.add_alert_handler(lambda a: alerts.append(a))
        
        # Phase 1: No drift (similar to reference)
        for _ in range(100):
            data = {
                'feature1': np.random.normal(0, 1),
                'feature2': np.random.normal(10, 2),
                'category': np.random.choice(['A', 'B', 'C'])
            }
            monitor.update(data)
        
        # Phase 2: Introduce drift in feature1
        for _ in range(100):
            data = {
                'feature1': np.random.normal(2, 1),  # Different mean
                'feature2': np.random.normal(10, 2),
                'category': np.random.choice(['A', 'B', 'C'])
            }
            monitor.update(data)
        
        # We should have received some alerts about feature1
        assert len(alerts) > 0
        assert any('feature1' in alert.message for alert in alerts)
        
        # No alerts for feature2 (no drift)
        assert not any('feature2' in alert.message for alert in alerts)
