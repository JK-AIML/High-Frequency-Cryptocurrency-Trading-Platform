"""
Comprehensive test suite for the drift monitoring system.

This module contains tests for all major components of the drift monitoring system,
including the UnifiedDriftDetector, DriftMonitor, alerting system, and integrations
with Spark and Flink.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import asyncio
import json

# Import the components to test
from tick_analysis.monitoring import (
    UnifiedDriftDetector,
    DriftMonitor,
    AlertRule,
    AlertSeverity,
    DriftType,
    DetectionMethod,
    DriftResult,
    DriftAlert
)

# Test data generators
class TestDataGenerator:
    """Helper class to generate test data for drift detection."""
    
    @staticmethod
    def create_reference_data(n_samples=1000, n_features=5, n_categories=3):
        """Create reference data with known distributions."""
        np.random.seed(42)
        data = {}
        
        # Numerical features with different distributions
        data['normal'] = np.random.normal(0, 1, n_samples)
        data['uniform'] = np.random.uniform(-5, 5, n_samples)
        data['bimodal'] = np.concatenate([
            np.random.normal(-2, 0.5, n_samples//2),
            np.random.normal(2, 0.5, n_samples//2)
        ])
        
        # Categorical features
        categories = [f'cat_{i}' for i in range(n_categories)]
        data['category'] = np.random.choice(categories, n_samples)
        
        # Add some noise features
        for i in range(n_features - 4):  # We already added 4 features
            data[f'noise_{i}'] = np.random.normal(0, 1, n_samples)
        
        return pd.DataFrame(data)
    
    @staticmethod
    def create_drifted_data(reference_df, drift_magnitude=1.0, n_samples=1000):
        """Create data with controlled drift from reference."""
        np.random.seed(43)  # Different seed for different data
        data = {}
        
        # Apply different types of drift to different features
        
        # Mean shift
        data['normal'] = np.random.normal(drift_magnitude, 1, n_samples)
        
        # Variance change
        data['uniform'] = np.random.uniform(-5 * (1 + drift_magnitude), 
                                          5 * (1 + drift_magnitude), 
                                          n_samples)
        
        # Distribution shape change
        if drift_magnitude > 0:
            data['bimodal'] = np.random.normal(0, 1 + drift_magnitude, n_samples)
        else:
            data['bimodal'] = np.concatenate([
                np.random.normal(-2, 0.5, n_samples//2),
                np.random.normal(2, 0.5, n_samples//2)
            ])
        
        # Categorical distribution shift
        categories = reference_df['category'].unique()
        probs = np.ones(len(categories)) / len(categories)
        if drift_magnitude > 0:
            probs[0] += drift_magnitude * 0.5
            probs = probs / probs.sum()
        data['category'] = np.random.choice(categories, n_samples, p=probs)
        
        # Copy noise features
        for col in reference_df.columns:
            if col.startswith('noise_'):
                data[col] = np.random.normal(0, 1, n_samples)
        
        return pd.DataFrame(data)

# Test cases
class TestUnifiedDriftDetector(unittest.TestCase):
    """Test the UnifiedDriftDetector class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data for all test methods."""
        cls.reference_data = TestDataGenerator.create_reference_data()
        cls.detector = UnifiedDriftDetector(
            reference_data=cls.reference_data,
            categorical_features=['category'],
            numerical_features=['normal', 'uniform', 'bimodal', 'noise_0'],
            alpha=0.05
        )
    
    def test_detect_drift_no_drift(self):
        """Test drift detection with no actual drift."""
        # Create current data with same distribution as reference
        current_data = TestDataGenerator.create_reference_data()
        
        # Test different detection methods
        for method in [
            DetectionMethod.KOLMOGOROV_SMIRNOV,
            DetectionMethod.JENSEN_SHANNON,
            DetectionMethod.WASSERSTEIN
        ]:
            with self.subTest(method=method):
                results = self.detector.detect_drift(
                    current_data=current_data,
                    method=method
                )
                
                # Check that no drift is detected in any feature
                for result in results:
                    self.assertFalse(result.is_drifted, 
                                   f"False positive drift detected with {method} in {result.feature}")
    
    def test_detect_drift_with_drift(self):
        """Test drift detection with actual drift."""
        # Create current data with drift
        current_data = TestDataGenerator.create_drifted_data(self.reference_data, drift_magnitude=1.0)
        
        # Test detection with KS test (should be sensitive to mean shift)
        results = self.detector.detect_drift(
            current_data=current_data,
            method=DetectionMethod.KOLMOGOROV_SMIRNOV
        )
        
        # Check that drift is detected in the 'normal' feature
        normal_result = next(r for r in results if r.feature == 'normal')
        self.assertTrue(normal_result.is_drifted, 
                       f"Expected drift in 'normal' feature not detected")
    
    def test_multivariate_drift(self):
        """Test multivariate drift detection."""
        # Create current data with multivariate drift
        current_data = TestDataGenerator.create_drifted_data(
            self.reference_data, 
            drift_magnitude=2.0
        )
        
        # Test multivariate detection
        result = self.detector.detect_multivariate_drift(
            current_data=current_data,
            method=DetectionMethod.MAHALANOBIS
        )
        
        self.assertIsInstance(result, DriftResult)
        self.assertEqual(result.drift_type, DriftType.MULTIVARIATE)
        self.assertTrue(result.is_drifted)


class TestDriftMonitor(unittest.TestCase):
    """Test the DriftMonitor class for real-time monitoring."""
    
    def setUp(self):
        """Set up test data and monitor for each test."""
        self.reference_data = TestDataGenerator.create_reference_data()
        self.monitor = DriftMonitor(
            reference_data=self.reference_data,
            window_size=100,
            min_samples=50,
            categorical_features=['category']
        )
        
        # Mock alert handler
        self.alert_handler = Mock()
        self.monitor.add_alert_handler(self.alert_handler)
    
    def test_initial_state(self):
        """Test the initial state of the monitor."""
        self.assertEqual(len(self.monitor.buffer), 0)
        self.assertEqual(len(self.monitor.alert_handlers), 1)
        self.assertEqual(len(self.monitor.alert_rules), 2)  # Default rules
    
    def test_update_no_drift(self):
        """Test updating the monitor with data that doesn't cause drift."""
        # Add reference-like data (no drift)
        for _ in range(200):
            sample = self.reference_data.sample(1).iloc[0].to_dict()
            self.monitor.update(sample)
        
        # No alerts should be triggered
        self.alert_handler.assert_not_called()
    
    def test_update_with_drift(self):
        """Test updating the monitor with data that causes drift."""
        # Add drifted data
        drifted_data = TestDataGenerator.create_drifted_data(
            self.reference_data, 
            drift_magnitude=2.0,
            n_samples=200
        )
        
        for _, row in drifted_data.iterrows():
            self.monitor.update(row.to_dict())
        
        # At least one alert should be triggered
        self.assertGreaterEqual(self.alert_handler.call_count, 1)
        
        # Check the alert content
        alert = self.alert_handler.call_args[0][0]
        self.assertIsInstance(alert, DriftAlert)
        self.assertIn(alert.severity, [AlertSeverity.WARNING, AlertSeverity.CRITICAL])
    
    @patch('tick_analysis.monitoring.drift_monitor.UnifiedDriftDetector')
    def test_alert_rule_triggering(self, mock_detector):
        """Test that alert rules trigger correctly."""
        # Set up mock detector
        mock_result = DriftResult(
            drift_type=DriftType.COVARIATE,
            detection_method=DetectionMethod.KOLMOGOROV_SMIRNOV,
            statistic=0.5,
            p_value=0.001,  # Should trigger critical alert
            is_drifted=True,
            feature='test_feature'
        )
        mock_detector.return_value.detect_drift.return_value = [mock_result]
        
        # Add a custom alert rule
        self.monitor.add_alert_rule(
            name="test_rule",
            condition=lambda r: r.p_value < 0.01,
            severity=AlertSeverity.CRITICAL,
            message_template="Test alert: {feature} has p={p_value:.4f}"
        )
        
        # Add some data to trigger detection
        for _ in range(100):
            self.monitor.update({'normal': 10.0})  # Clear outlier
        
        # Check that the alert was triggered
        self.alert_handler.assert_called()
        alert = self.alert_handler.call_args[0][0]
        self.assertEqual(alert.severity, AlertSeverity.CRITICAL)
        self.assertIn("test_feature", alert.message)


class TestAlertSystem(unittest.TestCase):
    """Test the alerting system."""
    
    def test_alert_rule_creation(self):
        """Test creating alert rules with different conditions."""
        # Test different conditions
        rules = [
            ("p_value_rule", lambda r: r.p_value < 0.05, AlertSeverity.WARNING),
            ("statistic_rule", lambda r: r.statistic > 0.5, AlertSeverity.CRITICAL),
            ("combined_rule", 
             lambda r: r.p_value < 0.01 and r.statistic > 0.5, 
             AlertSeverity.CRITICAL)
        ]
        
        for name, condition, severity in rules:
            with self.subTest(rule=name):
                rule = AlertRule(
                    name=name,
                    condition=condition,
                    severity=severity,
                    message_template="Test rule: {feature}"
                )
                
                # Test with a sample result
                result = DriftResult(
                    drift_type=DriftType.COVARIATE,
                    detection_method=DetectionMethod.KOLMOGOROV_SMIRNOV,
                    statistic=0.6,
                    p_value=0.001,
                    is_drifted=True,
                    feature="test_feature"
                )
                
                # Check if rule should trigger
                should_trigger = rule.should_trigger(result)
                self.assertTrue(should_trigger, f"Rule {name} should trigger")
                
                # Check alert creation
                alert = rule.create_alert(result)
                self.assertEqual(alert.severity, severity)
                self.assertIn("test_feature", alert.message)
    
    def test_alert_cooldown(self):
        """Test alert cooldown mechanism."""
        rule = AlertRule(
            name="test_cooldown",
            condition=lambda r: True,  # Always trigger
            severity=AlertSeverity.WARNING,
            message_template="Test cooldown",
            cooldown=60  # 1 minute cooldown
        )
        
        # Create a test result
        result = DriftResult(
            drift_type=DriftType.COVARIATE,
            detection_method=DetectionMethod.KOLMOGOROV_SMIRNOV,
            statistic=0.5,
            p_value=0.01,
            is_drifted=True,
            feature="test_feature"
        )
        
        # First trigger - should work
        self.assertTrue(rule.should_trigger(result))
        
        # Second trigger - should be in cooldown
        self.assertFalse(rule.should_trigger(result))
        
        # Simulate time passing
        rule.last_triggered = datetime.utcnow() - timedelta(seconds=61)
        
        # Should trigger again after cooldown
        self.assertTrue(rule.should_trigger(result))


class TestIntegration(unittest.TestCase):
    """Integration tests for the drift monitoring system."""
    
    def test_end_to_end(self):
        """Test the entire pipeline from data generation to alerting."""
        # Set up test data
        reference_data = TestDataGenerator.create_reference_data()
        
        # Set up monitor with custom alert rules
        monitor = DriftMonitor(
            reference_data=reference_data,
            window_size=100,
            min_samples=50,
            categorical_features=['category']
        )
        
        # Add alert handlers
        alerts = []
        def capture_alert(alert):
            alerts.append(alert)
        
        monitor.add_alert_handler(capture_alert)
        
        # Add a custom alert rule
        monitor.add_alert_rule(
            name="mean_shift",
            condition=lambda r: r.feature == 'normal' and r.p_value < 0.05,
            severity=AlertSeverity.WARNING,
            message_template="Mean shift in {feature}: p={p_value:.4f}"
        )
        
        # Phase 1: No drift
        for _ in range(100):
            sample = reference_data.sample(1).iloc[0].to_dict()
            monitor.update(sample)
        
        self.assertEqual(len(alerts), 0, "No alerts should be triggered in phase 1")
        
        # Phase 2: Introduce drift
        drifted_data = TestDataGenerator.create_drifted_data(
            reference_data,
            drift_magnitude=2.0,
            n_samples=200
        )
        
        for _, row in drifted_data.iterrows():
            monitor.update(row.to_dict())
        
        # Check that alerts were triggered
        self.assertGreater(len(alerts), 0, "Alerts should be triggered in phase 2")
        
        # Check that the alert is for the 'normal' feature
        normal_alerts = [a for a in alerts if 'normal' in a.message]
        self.assertGreater(len(normal_alerts), 0, 
                         "Expected alerts for 'normal' feature")


# Run the tests
if __name__ == '__main__':
    unittest.main()
