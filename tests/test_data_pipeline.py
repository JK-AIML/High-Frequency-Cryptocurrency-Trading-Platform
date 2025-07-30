"""
Test Data Pipeline

This module contains tests for the data pipeline functionality, including
the Dead Letter Queue and quality monitoring features.
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
import tempfile
import shutil

from tick_analysis.data_pipeline.pipeline import DataPipeline, PipelineConfig
from tick_analysis.data_pipeline.sources import FileSource
from tick_analysis.data_pipeline.storage import EnhancedParquetStorage
from tick_analysis.data_pipeline.dead_letter_queue import DeadLetterQueue
from tick_analysis.data_pipeline.quality_monitor import DataQualityMonitor
from tick_analysis.data_pipeline.recovery import PipelineRecovery
from tick_analysis.data_pipeline.monitoring import PipelineMonitor, Alert, HealthCheck

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_data(temp_dir):
    """Create sample data for testing."""
    # Create sample tick data
    data = []
    base_time = datetime.utcnow()
    
    for i in range(1000):
        data.append({
            'timestamp': (base_time + timedelta(seconds=i)).isoformat(),
            'symbol': 'BTC-USD',
            'price': 50000.0 + np.random.normal(0, 100),
            'volume': np.random.lognormal(0, 1),
            'bid': 50000.0 + np.random.normal(0, 50),
            'ask': 50000.0 + np.random.normal(0, 50)
        })
    
    # Save to parquet file
    df = pd.DataFrame(data)
    file_path = os.path.join(temp_dir, 'sample_ticks.parquet')
    df.to_parquet(file_path)
    
    return file_path

@pytest.fixture
def pipeline_config(temp_dir):
    """Create pipeline configuration for testing."""
    return PipelineConfig(
        source_type='file',
        source_config={
            'file_path': os.path.join(temp_dir, 'sample_ticks.parquet'),
            'format': 'parquet',
            'chunk_size': 100
        },
        storage_type='parquet',
        storage_config={
            'base_path': os.path.join(temp_dir, 'output'),
            'partition_cols': ['symbol', 'date']
        },
        dlq_enabled=True,
        dlq_storage_path=os.path.join(temp_dir, 'dlq'),
        quality_monitoring_enabled=True,
        quality_metrics_storage_path=os.path.join(temp_dir, 'quality_metrics'),
        recovery_enabled=True,
        monitoring_enabled=True
    )

@pytest.fixture
def test_data():
    """Create test tick data."""
    return [
        {
            'timestamp': '2024-01-01T00:00:00',
            'symbol': 'AAPL',
            'price': 150.0,
            'volume': 100
        },
        {
            'timestamp': '2024-01-01T00:00:01',
            'symbol': 'AAPL',
            'price': 151.0,
            'volume': 200
        }
    ]

@pytest.mark.asyncio
async def test_pipeline_basic_functionality(pipeline_config, sample_data):
    """Test basic pipeline functionality."""
    pipeline = DataPipeline(pipeline_config)
    
    try:
        # Start pipeline
        await pipeline.start()
        
        # Check if output files were created
        output_dir = os.path.join(pipeline_config.storage_config['base_path'], 'btc-usd_ticks')
        assert os.path.exists(output_dir)
        
        # Check if any files were written
        files = os.listdir(output_dir)
        assert len(files) > 0
        
        # Check metrics
        metrics = pipeline.get_metrics()
        assert metrics['processed'] > 0
        assert metrics['errors'] == 0
        
    finally:
        await pipeline.stop()

@pytest.mark.asyncio
async def test_dead_letter_queue(pipeline_config, temp_dir):
    """Test dead letter queue functionality."""
    # Create pipeline with failing storage
    pipeline_config.storage_config['base_path'] = '/invalid/path'
    pipeline = DataPipeline(pipeline_config)
    
    try:
        # Start pipeline
        await pipeline.start()
        
        # Check if DLQ files were created
        dlq_dir = pipeline_config.dlq_storage_path
        assert os.path.exists(dlq_dir)
        
        # Check if any failed batches were recorded
        files = os.listdir(dlq_dir)
        assert len(files) > 0
        
        # Check DLQ metrics
        dlq_metrics = pipeline.dlq.get_metrics()
        assert dlq_metrics['total_failures'] > 0
        
    finally:
        await pipeline.stop()

@pytest.mark.asyncio
async def test_quality_monitoring(pipeline_config, sample_data):
    """Test data quality monitoring functionality."""
    pipeline = DataPipeline(pipeline_config)
    
    try:
        # Start pipeline
        await pipeline.start()
        
        # Check if quality metrics were created
        metrics_dir = pipeline_config.quality_metrics_storage_path
        assert os.path.exists(metrics_dir)
        
        # Check if any metrics were recorded
        files = os.listdir(metrics_dir)
        assert len(files) > 0
        
        # Check quality metrics
        quality_metrics = pipeline.quality_monitor.get_metrics()
        assert quality_metrics is not None
        
    finally:
        await pipeline.stop()

@pytest.mark.asyncio
async def test_data_drift_detection(pipeline_config, sample_data):
    """Test data drift detection functionality."""
    pipeline = DataPipeline(pipeline_config)
    
    try:
        # Create reference data with different distribution
        reference_data = pd.DataFrame({
            'timestamp': [datetime.utcnow().isoformat() for _ in range(1000)],
            'symbol': ['BTC-USD'] * 1000,
            'price': 60000.0 + np.random.normal(0, 200),  # Different mean and variance
            'volume': np.random.lognormal(0, 2),  # Different distribution
            'bid': 60000.0 + np.random.normal(0, 100),
            'ask': 60000.0 + np.random.normal(0, 100)
        })
        
        # Set reference data
        pipeline.quality_monitor.set_reference_data(reference_data)
        
        # Start pipeline
        await pipeline.start()
        
        # Check if drift was detected
        quality_metrics = pipeline.quality_monitor.get_metrics()
        drift_metrics = {
            name: metrics[-1].value
            for name, metrics in quality_metrics.items()
            if 'drift' in name and metrics[-1].severity in ['ERROR', 'CRITICAL']
        }
        
        assert len(drift_metrics) > 0
        
    finally:
        await pipeline.stop()

@pytest.mark.asyncio
async def test_anomaly_detection(pipeline_config, sample_data):
    """Test anomaly detection functionality."""
    pipeline = DataPipeline(pipeline_config)
    
    try:
        # Create data with anomalies
        df = pd.read_parquet(sample_data)
        df.loc[100:110, 'price'] = 1000000.0  # Add some extreme values
        df.to_parquet(sample_data)
        
        # Start pipeline
        await pipeline.start()
        
        # Check if anomalies were detected
        quality_metrics = pipeline.quality_monitor.get_metrics()
        anomaly_metrics = {
            name: metrics[-1].value
            for name, metrics in quality_metrics.items()
            if 'anomaly' in name and metrics[-1].severity in ['ERROR', 'CRITICAL']
        }
        
        assert len(anomaly_metrics) > 0
        
    finally:
        await pipeline.stop()

@pytest.mark.asyncio
async def test_pipeline_recovery(pipeline_config, sample_data):
    """Test pipeline recovery after failure."""
    pipeline = DataPipeline(pipeline_config)
    
    try:
        # Start pipeline
        await pipeline.start()
        
        # Simulate failure
        await pipeline.stop()
        
        # Restart pipeline
        await pipeline.start()
        
        # Check if pipeline recovered
        metrics = pipeline.get_metrics()
        assert metrics['processed'] > 0
        
    finally:
        await pipeline.stop()

@pytest.mark.asyncio
async def test_metrics_collection(pipeline_config, sample_data):
    """Test metrics collection functionality."""
    pipeline = DataPipeline(pipeline_config)
    
    try:
        # Start pipeline
        await pipeline.start()
        
        # Wait for some metrics to be collected
        await asyncio.sleep(pipeline_config.metrics_interval + 1)
        
        # Check metrics
        metrics = pipeline.get_metrics()
        assert 'processed' in metrics
        assert 'errors' in metrics
        assert 'batches_processed' in metrics
        assert 'dlq_metrics' in metrics
        assert 'quality_metrics' in metrics
        
    finally:
        await pipeline.stop()

@pytest.mark.asyncio
async def test_pipeline_initialization(pipeline_config):
    """Test pipeline initialization."""
    pipeline = DataPipeline(pipeline_config)
    
    assert pipeline.pipeline_id is not None
    assert pipeline.source is not None
    assert len(pipeline.processors) == 2
    assert pipeline.storage is not None
    assert pipeline.dlq is not None
    assert pipeline.quality_monitor is not None
    assert pipeline.recovery is not None
    assert pipeline.monitor is not None

@pytest.mark.asyncio
async def test_pipeline_start_stop(pipeline_config):
    """Test pipeline start and stop."""
    pipeline = DataPipeline(pipeline_config)
    
    await pipeline.start()
    assert pipeline._running is True
    assert pipeline._processing_task is not None
    assert pipeline._metrics_task is not None
    
    await pipeline.stop()
    assert pipeline._running is False
    assert pipeline._processing_task.cancelled()
    assert pipeline._metrics_task.cancelled()

@pytest.mark.asyncio
async def test_dead_letter_queue_new(pipeline_config):
    """Test Dead Letter Queue functionality."""
    dlq = DeadLetterQueue(
        pipeline_id='test_pipeline',
        max_retries=3,
        retry_delay=0.1
    )
    
    await dlq.start()
    
    # Add failed batch
    failed_batch = [{'data': 'test'}]
    error = 'Test error'
    await dlq.add_failed_batch(failed_batch, error)
    
    # Verify batch was added
    assert len(dlq.failed_batches) == 1
    assert dlq.failed_batches[0]['error'] == error
    
    # Process failed batch
    await dlq.process_failed_batches()
    
    # Verify batch was processed
    assert len(dlq.failed_batches) == 0
    
    await dlq.stop()

@pytest.mark.asyncio
async def test_quality_monitor_new(pipeline_config):
    """Test Quality Monitor functionality."""
    monitor = DataQualityMonitor(
        pipeline_id='test_pipeline',
        storage_path='./data/quality'
    )
    
    await monitor.start()
    
    # Record metrics
    test_data = [
        {'price': 100.0, 'volume': 1000},
        {'price': 101.0, 'volume': 2000}
    ]
    await monitor.record_metrics(test_data)
    
    # Verify metrics were recorded
    metrics = monitor.get_metrics()
    assert 'price' in metrics
    assert 'volume' in metrics
    assert metrics['price']['mean'] == 100.5
    assert metrics['volume']['mean'] == 1500
    
    await monitor.stop()

@pytest.mark.asyncio
async def test_pipeline_recovery_new(pipeline_config):
    """Test Pipeline Recovery functionality."""
    recovery = PipelineRecovery(
        pipeline_id='test_pipeline',
        storage_path='./data/recovery'
    )
    
    await recovery.start()
    
    # Update state
    test_state = {
        'last_processed_record': {'id': 1},
        'metrics': {'processed_records': 100}
    }
    await recovery.update_state(**test_state)
    
    # Verify state was saved
    state = await recovery.load_checkpoint()
    assert state is not None
    assert state.last_processed_record == test_state['last_processed_record']
    assert state.metrics == test_state['metrics']
    
    await recovery.stop()

@pytest.mark.asyncio
async def test_pipeline_monitor_new(pipeline_config):
    """Test Pipeline Monitor functionality."""
    monitor = PipelineMonitor(
        pipeline_id='test_pipeline',
        metrics_port=9090
    )
    
    await monitor.start()
    
    # Record metrics
    test_metrics = {
        'processed_records': 100,
        'processing_errors': {'error': 1},
        'processing_latency': 0.1,
        'batch_size': 50,
        'queue_size': 25
    }
    await monitor.record_metrics(test_metrics)
    
    # Create alert
    alert = await monitor.create_alert(
        name='test_alert',
        severity='warning',
        message='Test alert',
        metadata={'test': True}
    )
    
    assert alert.name == 'test_alert'
    assert alert.severity == 'warning'
    assert alert.message == 'Test alert'
    assert alert.metadata['test'] is True
    
    # Resolve alert
    resolved_alert = await monitor.resolve_alert('test_alert')
    assert resolved_alert.resolved is True
    assert resolved_alert.resolved_at is not None
    
    # Check health
    health_checks = await monitor.check_health()
    assert 'memory' in health_checks
    assert 'cpu' in health_checks
    assert 'disk' in health_checks
    
    await monitor.stop()

@pytest.mark.asyncio
async def test_pipeline_integration(pipeline_config, test_data):
    """Test full pipeline integration."""
    # Create test data file
    os.makedirs('./data', exist_ok=True)
    with open('./data/test_ticks.csv', 'w') as f:
        for record in test_data:
            f.write(json.dumps(record) + '\n')
    
    # Initialize pipeline
    pipeline = DataPipeline(pipeline_config)
    await pipeline.start()
    
    # Wait for processing
    await asyncio.sleep(2)
    
    # Check pipeline info
    info = pipeline.get_pipeline_info()
    assert info['status'] == 'running'
    assert info['metrics']['processed_records'] > 0
    
    # Stop pipeline
    await pipeline.stop()
    
    # Clean up
    os.remove('./data/test_ticks.csv')
    os.rmdir('./data')

@pytest.mark.asyncio
async def test_pipeline_recovery_integration(pipeline_config):
    """Test pipeline recovery integration."""
    pipeline = DataPipeline(pipeline_config)
    
    # Start pipeline
    await pipeline.start()
    
    # Simulate failure
    await pipeline.stop()
    
    # Recover pipeline
    await pipeline.recover()
    
    # Verify recovery
    assert pipeline._running is True
    assert pipeline._processing_task is not None
    assert pipeline._metrics_task is not None
    
    # Stop pipeline
    await pipeline.stop()

@pytest.mark.asyncio
async def test_pipeline_monitoring_integration(pipeline_config):
    """Test pipeline monitoring integration."""
    pipeline = DataPipeline(pipeline_config)
    
    # Start pipeline
    await pipeline.start()
    
    # Wait for monitoring
    await asyncio.sleep(2)
    
    # Check monitoring info
    monitoring_info = pipeline.monitor.get_monitoring_info()
    assert monitoring_info['pipeline_id'] == pipeline.pipeline_id
    assert monitoring_info['uptime'] > 0
    
    # Stop pipeline
    await pipeline.stop()

@pytest.mark.asyncio
async def test_pipeline_error_handling(pipeline_config):
    """Test pipeline error handling."""
    pipeline = DataPipeline(pipeline_config)
    
    # Start pipeline
    await pipeline.start()
    
    # Simulate error in source
    pipeline.source.read = lambda: asyncio.sleep(0.1) or 1/0
    
    # Wait for error handling
    await asyncio.sleep(2)
    
    # Verify error was handled
    assert pipeline.metrics['processing_errors'].get('ZeroDivisionError', 0) > 0
    
    # Stop pipeline
    await pipeline.stop()

@pytest.mark.asyncio
async def test_pipeline_cleanup(pipeline_config):
    """Test pipeline cleanup."""
    pipeline = DataPipeline(pipeline_config)
    
    # Start pipeline
    await pipeline.start()
    
    # Stop pipeline
    await pipeline.stop()
    
    # Verify cleanup
    assert pipeline._running is False
    assert pipeline._processing_task.cancelled()
    assert pipeline._metrics_task.cancelled()
    
    # Clean up test directories
    for path in ['./data/quality', './data/recovery']:
        if os.path.exists(path):
            os.rmdir(path) 