"""
Tests for the streaming processors (Flink and Spark).
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json
import tempfile
import os

from tick_analysis.streaming.flink_processor import FlinkStreamProcessor, WindowConfig, ProcessingMode
from tick_analysis.streaming.spark_processor import SparkStreamProcessor as SparkProcessor

# Test data
def generate_test_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate test market data."""
    np.random.seed(42)
    timestamps = pd.date_range(
        start=datetime.utcnow() - timedelta(minutes=60),
        periods=n_samples,
        freq='s'
    )
    
    return pd.DataFrame({
        'symbol': np.random.choice(['AAPL', 'GOOGL', 'MSFT', 'AMZN'], size=n_samples),
        'price': 100 + np.cumsum(np.random.randn(n_samples) * 0.1),
        'volume': np.random.randint(100, 1000, size=n_samples),
        'exchange': 'NASDAQ',
        'timestamp': timestamps,
        'trade_id': [f'trade_{i}' for i in range(n_samples)],
        'side': np.random.choice(['BUY', 'SELL'], size=n_samples),
        'conditions': [['NORMAL']] * n_samples
    })

# Fixtures
@pytest.fixture
def flink_config() -> Dict[str, Any]:
    """Return a Flink processor configuration."""
    return {
        'flink': {
            'parallelism': 2,
            'checkpoint_interval': 5000,
            'checkpoint_timeout': 30000
        },
        'kafka': {
            'bootstrap_servers': 'localhost:9092',
            'group_id': 'test-flink-consumer',
            'auto_offset_reset': 'earliest',
            'sink_parallelism': 2,
            'sink_flush_interval': '1s',
            'sink_buffer_max_rows': 1000,
            'delivery_guarantee': 'at-least-once'
        },
        'enable_drift_detection': False
    }

@pytest.fixture
def spark_config() -> Dict[str, Any]:
    """Return a Spark processor configuration."""
    return {
        'spark': {
            'shuffle_partitions': 2
        },
        'kafka': {
            'bootstrap_servers': 'localhost:9092',
            'group_id': 'test-spark-consumer'
        },
        'checkpoint_location': os.path.join(tempfile.gettempdir(), 'spark-checkpoints'),
        'enable_drift_detection': False
    }

@pytest.fixture
def test_data() -> pd.DataFrame:
    """Generate test market data."""
    return generate_test_data(1000)

# Tests
class TestFlinkProcessor:
    """Test cases for FlinkStreamProcessor."""
    
    def test_initialization(self, flink_config):
        """Test Flink processor initialization."""
        processor = FlinkStreamProcessor(
            config=flink_config,
            processing_mode=ProcessingMode.STREAMING
        )
        assert processor is not None
        assert processor.env is not None
        assert processor.table_env is not None
    
    def test_window_config(self):
        """Test window configuration."""
        window_cfg = WindowConfig(
            window_size=300,  # 5 minutes
            slide_size=60,   # 1 minute
            window_type='sliding'
        )
        assert window_cfg.window_size == 300
        assert window_cfg.slide_size == 60
        assert window_cfg.window_type == 'sliding'
    
    def test_process_tick_data(self, flink_config, test_data):
        """Test tick data processing."""
        # This is a basic test - in a real scenario, you'd mock Kafka
        processor = FlinkStreamProcessor(
            config=flink_config,
            processing_mode=ProcessingMode.STREAMING
        )
        
        # Test would normally process data here
        # For now, just verify the method exists and is callable
        assert hasattr(processor, 'process_tick_data')
        assert callable(processor.process_tick_data)


class TestSparkProcessor:
    """Test cases for SparkStreamProcessor."""
    
    def test_initialization(self, spark_config):
        """Test Spark processor initialization."""
        processor = SparkProcessor(
            config=spark_config,
            app_name="test-spark-processor"
        )
        assert processor is not None
        assert processor.spark is not None
        assert processor.checkpoint_location == spark_config['checkpoint_location']
    
    def test_window_config(self):
        """Test window configuration."""
        window_cfg = WindowConfig(
            window_duration="5 minutes",
            slide_duration="1 minute",
            watermark_delay="10 minutes"
        )
        assert window_cfg.window_duration == "5 minutes"
        assert window_cfg.slide_duration == "1 minute"
        assert window_cfg.watermark_delay == "10 minutes"
    
    def test_process_tick_data(self, spark_config, test_data):
        """Test tick data processing."""
        # This is a basic test - in a real scenario, you'd mock SparkSession
        processor = SparkProcessor(
            config=spark_config,
            app_name="test-spark-processor"
        )
        
        # Test would normally process data here
        # For now, just verify the method exists and is callable
        assert hasattr(processor, 'process_tick_data')
        assert callable(processor.process_tick_data)


def test_integration(flink_config, spark_config, test_data):
    """Test integration between Flink and Spark processors."""
    # This would test end-to-end flow from Flink to Spark
    # For now, just verify both processors can be created with test configs
    flink_processor = FlinkStreamProcessor(
        config=flink_config,
        processing_mode=ProcessingMode.STREAMING
    )
    
    spark_processor = SparkProcessor(
        config=spark_config,
        app_name="test-integration"
    )
    
    assert flink_processor is not None
    assert spark_processor is not None


if __name__ == "__main__":
    pytest.main(["-v", "test_streaming_processors.py"])
