"""
Test suite for data processing components.
"""

import pytest
import asyncio
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.tick_analysis.data_pipeline.processing import (
    StreamConfig,
    FeatureConfig,
    StreamProcessor,
    FeatureStore,
    DriftDetector,
    ProcessingConfig,
    PolarsProcessor
)
import polars as pl

@pytest.fixture
def test_data():
    """Generate test data."""
    base_time = datetime.utcnow()
    data = []
    
    for i in range(100):
        record = {
            'timestamp': base_time + timedelta(minutes=i),
            'symbol': f'SYMBOL_{i % 5}',
            'price': 100.0 + i * 0.1,
            'volume': 1000 + i * 10,
            'bid': 99.9 + i * 0.1,
            'ask': 100.1 + i * 0.1
        }
        data.append(record)
    
    return data

@pytest.fixture
def stream_config():
    """Create stream processing configuration."""
    return StreamConfig(
        window_size=10,
        slide_interval=5,
        watermark_delay=2,
        allowed_lateness=5
    )

@pytest.fixture
def feature_config():
    """Create feature configuration."""
    return FeatureConfig(
        name='price_feature',
        description='Price feature for tick data',
        data_type='float',
        tags=['price', 'tick_data']
    )

@pytest.fixture
def processing_config():
    """Create processing configuration."""
    return ProcessingConfig(
        batch_size=1000,
        num_threads=4,
        window_size=100,
        min_periods=20,
        processing_timeout=30
    )

@pytest.mark.asyncio
async def test_stream_processor(stream_config, test_data):
    """Test stream processing."""
    # Initialize processor
    processor = StreamProcessor(stream_config)
    
    # Add window handler
    processed_windows = []
    async def handler(window_data, key):
        processed_windows.append((key, window_data))
    processor.add_handler(handler)
    
    # Process records
    for record in test_data:
        await processor.process_record(record, record['symbol'])
    
    # Verify window processing
    assert len(processed_windows) > 0
    for key, window in processed_windows:
        assert len(window) == stream_config.window_size
    
    # Verify state
    state = processor.get_state()
    assert len(state) > 0
    for key in state:
        assert state[key]['count'] > 0
    
    # Verify checkpoints
    assert len(processor.checkpoints) > 0

@pytest.mark.asyncio
async def test_feature_store(feature_config, test_data):
    """Test feature store."""
    # Initialize feature store
    store = FeatureStore('./test_features')
    
    try:
        # Register feature
        store.register_feature(feature_config)
        
        # Store feature data
        await store.store_feature(feature_config.name, test_data)
        
        # Get feature data
        data = await store.get_feature(feature_config.name)
        assert len(data) == len(test_data)
        
        # Test filtering
        filtered_data = await store.get_feature(
            feature_config.name,
            filters={'symbol': 'SYMBOL_0'}
        )
        assert len(filtered_data) > 0
        
        # Test feature scaling
        store.fit_scaler(feature_config.name, data[['price']])
        scaled_data = store.transform_feature(feature_config.name, data[['price']])
        assert len(scaled_data) == len(data)
        
        # Test feature listing
        features = store.list_features()
        assert len(features) == 1
        assert features[0]['name'] == feature_config.name
        
        # Test metadata retrieval
        metadata = store.get_feature_metadata(feature_config.name)
        assert metadata['name'] == feature_config.name
        
    finally:
        # Cleanup
        import shutil
        shutil.rmtree('./test_features')

@pytest.mark.asyncio
async def test_drift_detection(test_data):
    """Test drift detection."""
    # Initialize drift detector
    detector = DriftDetector(window_size=20, threshold=0.05)
    
    # Convert test data to DataFrame
    df = pd.DataFrame(test_data)
    
    # Update reference data
    detector.update_reference('price', df['price'])
    
    # Generate drifted data
    drifted_data = df['price'] * 1.1  # 10% increase
    
    # Detect drift
    result = detector.detect_drift('price', drifted_data)
    assert result['drift_detected']
    assert result['drift_score'] > 0
    
    # Test drift scores
    scores = detector.get_drift_scores()
    assert 'price' in scores
    assert scores['price'] > 0
    
    # Test reset
    detector.reset('price')
    assert 'price' not in detector.reference_data
    assert 'price' not in detector.drift_scores

@pytest.mark.asyncio
async def test_stream_processor_error_handling(stream_config):
    """Test stream processor error handling."""
    processor = StreamProcessor(stream_config)
    
    # Test invalid window size
    with pytest.raises(ValueError):
        processor = StreamProcessor(StreamConfig(
            window_size=0,
            slide_interval=5,
            watermark_delay=2,
            allowed_lateness=5
        ))
    
    # Test invalid slide interval
    with pytest.raises(ValueError):
        processor = StreamProcessor(StreamConfig(
            window_size=10,
            slide_interval=11,
            watermark_delay=2,
            allowed_lateness=5
        ))

@pytest.mark.asyncio
async def test_feature_store_error_handling(feature_config):
    """Test feature store error handling."""
    store = FeatureStore('./test_features')
    
    try:
        # Test duplicate feature registration
        store.register_feature(feature_config)
        with pytest.raises(ValueError):
            store.register_feature(feature_config)
        
        # Test non-existent feature
        with pytest.raises(ValueError):
            await store.get_feature('non_existent')
        
        # Test invalid data type
        with pytest.raises(ValueError):
            await store.store_feature(feature_config.name, 'invalid_data')
        
    finally:
        # Cleanup
        import shutil
        shutil.rmtree('./test_features')

@pytest.mark.asyncio
async def test_drift_detection_error_handling():
    """Test drift detection error handling."""
    detector = DriftDetector(window_size=20)
    
    # Test insufficient data
    with pytest.raises(ValueError):
        detector.update_reference('price', pd.Series([1, 2, 3]))
    
    # Test non-existent feature
    with pytest.raises(ValueError):
        detector.detect_drift('non_existent', pd.Series(range(100)))

@pytest.mark.asyncio
async def test_stream_processor_checkpointing(stream_config, test_data):
    """Test stream processor checkpointing."""
    processor = StreamProcessor(stream_config)
    
    # Process some records
    for record in test_data[:50]:
        await processor.process_record(record, record['symbol'])
    
    # Create checkpoint
    checkpoint_id = f"checkpoint_{len(processor.checkpoints)}"
    processor.checkpoints[checkpoint_id] = {
        'timestamp': datetime.utcnow(),
        'state': dict(processor.state),
        'windows': {k: len(v) for k, v in processor.windows.items()}
    }
    
    # Process more records
    for record in test_data[50:]:
        await processor.process_record(record, record['symbol'])
    
    # Restore checkpoint
    await processor.restore_checkpoint(checkpoint_id)
    
    # Verify state restoration
    assert len(processor.state) > 0
    assert len(processor.windows) > 0

@pytest.mark.asyncio
async def test_feature_store_scaling(feature_config, test_data):
    """Test feature store scaling."""
    store = FeatureStore('./test_features')
    
    try:
        # Register and store feature
        store.register_feature(feature_config)
        await store.store_feature(feature_config.name, test_data)
        
        # Get feature data
        data = await store.get_feature(feature_config.name)
        
        # Test scaling
        store.fit_scaler(feature_config.name, data[['price']])
        scaled_data = store.transform_feature(feature_config.name, data[['price']])
        
        # Verify scaling
        assert scaled_data['price'].mean() == pytest.approx(0, abs=1e-10)
        assert scaled_data['price'].std() == pytest.approx(1, abs=1e-10)
        
    finally:
        # Cleanup
        import shutil
        shutil.rmtree('./test_features')

@pytest.mark.asyncio
async def test_batch_processing(processing_config, test_data):
    """Test batch processing functionality."""
    # Initialize processor
    processor = PolarsProcessor(processing_config)
    
    try:
        # Process batch
        processed_df = await processor.process_batch(test_data)
        
        # Check DataFrame structure
        assert isinstance(processed_df, pl.DataFrame)
        assert len(processed_df) == len(test_data)
        
        # Check basic columns
        assert 'timestamp' in processed_df.columns
        assert 'price' in processed_df.columns
        assert 'volume' in processed_df.columns
        
        # Check calculated columns
        assert 'price_change' in processed_df.columns
        assert 'price_return' in processed_df.columns
        assert 'volume_ma' in processed_df.columns
        assert 'volatility' in processed_df.columns
        assert 'vwap' in processed_df.columns
        
        # Check technical indicators
        assert 'sma_20' in processed_df.columns
        assert 'sma_50' in processed_df.columns
        assert 'sma_200' in processed_df.columns
        assert 'rsi' in processed_df.columns
        assert 'macd' in processed_df.columns
        assert 'macd_signal' in processed_df.columns
        assert 'macd_histogram' in processed_df.columns
        assert 'bb_middle' in processed_df.columns
        assert 'bb_upper' in processed_df.columns
        assert 'bb_lower' in processed_df.columns
        
    finally:
        # Cleanup
        await processor.stop()

@pytest.mark.asyncio
async def test_stream_processing(processing_config, test_data):
    """Test stream processing functionality."""
    # Initialize processor
    processor = PolarsProcessor(processing_config)
    
    try:
        # Create input queue
        input_queue = asyncio.Queue()
        
        # Start stream processing
        output_queue = await processor.process_stream(input_queue)
        
        # Add data to input queue
        for i in range(0, len(test_data), processing_config.batch_size):
            batch = test_data[i:i + processing_config.batch_size]
            await input_queue.put(batch)
        
        # Get processed data
        processed_batches = []
        for _ in range(len(test_data) // processing_config.batch_size):
            processed_batch = await output_queue.get()
            processed_batches.append(processed_batch)
        
        # Check results
        assert len(processed_batches) > 0
        for batch in processed_batches:
            assert isinstance(batch, pl.DataFrame)
            assert len(batch) > 0
            
    finally:
        # Cleanup
        await processor.stop()

@pytest.mark.asyncio
async def test_technical_indicators(processing_config, test_data):
    """Test technical indicators calculation."""
    # Initialize processor
    processor = PolarsProcessor(processing_config)
    
    try:
        # Process batch
        processed_df = await processor.process_batch(test_data)
        
        # Check RSI
        assert 'rsi' in processed_df.columns
        rsi_values = processed_df['rsi'].to_numpy()
        assert np.all((rsi_values >= 0) & (rsi_values <= 100))
        
        # Check MACD
        assert 'macd' in processed_df.columns
        assert 'macd_signal' in processed_df.columns
        assert 'macd_histogram' in processed_df.columns
        
        # Check Bollinger Bands
        assert 'bb_middle' in processed_df.columns
        assert 'bb_upper' in processed_df.columns
        assert 'bb_lower' in processed_df.columns
        
        # Verify Bollinger Bands relationship
        bb_middle = processed_df['bb_middle'].to_numpy()
        bb_upper = processed_df['bb_upper'].to_numpy()
        bb_lower = processed_df['bb_lower'].to_numpy()
        
        assert np.all(bb_upper >= bb_middle)
        assert np.all(bb_middle >= bb_lower)
        
    finally:
        # Cleanup
        await processor.stop()

@pytest.mark.asyncio
async def test_error_handling(processing_config):
    """Test error handling."""
    # Initialize processor
    processor = PolarsProcessor(processing_config)
    
    try:
        # Test with invalid data
        invalid_data = [
            {'timestamp': 'invalid', 'price': 'invalid', 'volume': 'invalid'}
        ]
        
        with pytest.raises(Exception):
            await processor.process_batch(invalid_data)
        
    finally:
        # Cleanup
        await processor.stop()

@pytest.mark.asyncio
async def test_concurrent_processing(processing_config, test_data):
    """Test concurrent processing capabilities."""
    # Initialize processor
    processor = PolarsProcessor(processing_config)
    
    try:
        # Create multiple batches
        batches = [
            test_data[i:i + processing_config.batch_size]
            for i in range(0, len(test_data), processing_config.batch_size)
        ]
        
        # Process batches concurrently
        tasks = [
            processor.process_batch(batch)
            for batch in batches
        ]
        
        # Wait for all tasks to complete
        processed_batches = await asyncio.gather(*tasks)
        
        # Check results
        assert len(processed_batches) == len(batches)
        for batch in processed_batches:
            assert isinstance(batch, pl.DataFrame)
            assert len(batch) > 0
            
    finally:
        # Cleanup
        await processor.stop()

@pytest.mark.asyncio
async def test_performance(processing_config, test_data):
    """Test processing performance."""
    # Initialize processor
    processor = PolarsProcessor(processing_config)
    
    try:
        # Measure processing time
        start_time = asyncio.get_event_loop().time()
        
        # Process batch
        processed_df = await processor.process_batch(test_data)
        
        end_time = asyncio.get_event_loop().time()
        processing_time = end_time - start_time
        
        # Check performance
        assert processing_time < processing_config.processing_timeout
        assert len(processed_df) == len(test_data)
        
    finally:
        # Cleanup
        await processor.stop() 