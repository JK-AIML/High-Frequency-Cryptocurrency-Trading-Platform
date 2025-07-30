"""
Test suite for Rust wrapper implementation.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from src.tick_analysis.data_pipeline.rust_wrapper import (
    RustProcessingConfig,
    RustProcessor
)

@pytest.fixture
def rust_config():
    """Create Rust processing configuration."""
    return RustProcessingConfig(
        window_size=100,
        min_periods=20,
        use_rust=True
    )

@pytest.fixture
def test_data():
    """Generate test data."""
    # Generate timestamps
    timestamps = [
        datetime.utcnow() + timedelta(seconds=i)
        for i in range(1000)
    ]
    
    # Generate prices with some trend and noise
    base_price = 100.0
    trend = np.linspace(0, 10, 1000)
    noise = np.random.normal(0, 1, 1000)
    prices = base_price + trend + noise
    
    # Generate volumes
    volumes = np.random.lognormal(5, 1, 1000)
    
    return {
        'prices': prices,
        'volumes': volumes
    }

def test_processor_initialization(rust_config):
    """Test processor initialization."""
    processor = RustProcessor(rust_config)
    assert processor.config == rust_config
    assert processor.use_rust == (rust_config.use_rust and processor.RUST_AVAILABLE)

def test_data_processing(rust_config, test_data):
    """Test data processing functionality."""
    processor = RustProcessor(rust_config)
    
    # Process data
    results = processor.process_data(
        test_data['prices'],
        test_data['volumes']
    )
    
    # Check results structure
    assert isinstance(results, dict)
    assert 'sma' in results
    assert 'vwap' in results
    assert 'statistics' in results
    
    # Check technical indicators
    assert isinstance(results['sma'], np.ndarray)
    assert isinstance(results['vwap'], np.ndarray)
    
    # Check statistics
    stats = results['statistics']
    assert 'mean' in stats
    assert 'std' in stats
    assert 'min' in stats
    assert 'max' in stats

def test_custom_indicators(rust_config, test_data):
    """Test custom indicators calculation."""
    processor = RustProcessor(rust_config)
    
    # Create test data matrix
    data = np.column_stack([
        test_data['prices'],
        test_data['volumes']
    ])
    
    # Calculate custom indicators
    results = processor.calculate_custom_indicators(data)
    
    # Check results structure
    assert isinstance(results, dict)
    assert 'statistics' in results
    assert 'correlation' in results
    
    # Check statistics
    stats = results['statistics']
    assert 'mean' in stats
    assert 'std' in stats
    assert 'min' in stats
    assert 'max' in stats
    
    # Check correlation matrix
    correlation = results['correlation']
    assert isinstance(correlation, np.ndarray)
    assert correlation.shape == (2, 2)  # 2x2 matrix for 2 features

def test_python_fallback(rust_config, test_data):
    """Test Python fallback implementation."""
    # Create config with Rust disabled
    config = RustProcessingConfig(
        window_size=100,
        min_periods=20,
        use_rust=False
    )
    
    processor = RustProcessor(config)
    
    # Process data
    results = processor.process_data(
        test_data['prices'],
        test_data['volumes']
    )
    
    # Check results structure
    assert isinstance(results, dict)
    assert 'sma' in results
    assert 'vwap' in results
    assert 'statistics' in stats
    
    # Check technical indicators
    assert isinstance(results['sma'], np.ndarray)
    assert isinstance(results['vwap'], np.ndarray)
    
    # Check statistics
    stats = results['statistics']
    assert 'mean' in stats
    assert 'std' in stats
    assert 'min' in stats
    assert 'max' in stats

def test_error_handling(rust_config):
    """Test error handling."""
    processor = RustProcessor(rust_config)
    
    # Test with invalid data
    invalid_prices = np.array(['invalid'])
    invalid_volumes = np.array(['invalid'])
    
    with pytest.raises(Exception):
        processor.process_data(invalid_prices, invalid_volumes)
    
    # Test with empty data
    empty_prices = np.array([])
    empty_volumes = np.array([])
    
    with pytest.raises(Exception):
        processor.process_data(empty_prices, empty_volumes)

def test_performance(rust_config, test_data):
    """Test processing performance."""
    processor = RustProcessor(rust_config)
    
    # Measure processing time
    import time
    start_time = time.time()
    
    # Process data
    results = processor.process_data(
        test_data['prices'],
        test_data['volumes']
    )
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Check performance
    assert processing_time < 1.0  # Should process 1000 points in less than 1 second
    assert len(results['sma']) == len(test_data['prices']) - rust_config.window_size + 1 