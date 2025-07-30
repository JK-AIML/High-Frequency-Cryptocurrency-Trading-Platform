"""
Test suite for enhanced storage system.
"""

import pytest
import asyncio
import os
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from src.tick_analysis.data_pipeline.storage import (
    StorageType,
    PartitionStrategy,
    IndexType,
    PartitionConfig,
    IndexConfig,
    RetentionConfig,
    EnhancedInfluxDBStorage,
    EnhancedParquetStorage
)

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
def partition_config():
    """Create partition configuration."""
    return PartitionConfig(
        strategy=PartitionStrategy.TIME,
        field='timestamp',
        interval='hour'
    )

@pytest.fixture
def index_config():
    """Create index configuration."""
    return IndexConfig(
        type=IndexType.BTREE,
        fields=['symbol', 'price'],
        unique=False
    )

@pytest.fixture
def retention_config():
    """Create retention configuration."""
    return RetentionConfig(
        max_age=timedelta(days=7),
        max_size=1024 * 1024,  # 1MB
        compression=True,
        archive=True,
        archive_path='./archive'
    )

@pytest.mark.asyncio
async def test_enhanced_parquet_storage(test_data, partition_config, index_config, retention_config):
    """Test enhanced Parquet storage."""
    # Initialize storage
    storage = EnhancedParquetStorage(
        storage_path='./test_storage',
        partition_config=partition_config,
        index_config=index_config,
        retention_config=retention_config
    )
    
    # Store data
    await storage.store(test_data)
    
    # Query data
    results = await storage.query()
    assert len(results) == len(test_data)
    
    # Query with partition key
    partition_key = pd.to_datetime(test_data[0]['timestamp']).strftime('%Y-%m-%d-%H')
    results = await storage.query(partition_key=partition_key)
    assert len(results) > 0
    
    # Query with filters
    results = await storage.query(filters={'symbol': 'SYMBOL_0'})
    assert len(results) > 0
    
    # Cleanup
    import shutil
    shutil.rmtree('./test_storage')
    if os.path.exists('./archive'):
        shutil.rmtree('./archive')

@pytest.mark.asyncio
async def test_enhanced_influxdb_storage(test_data, partition_config, index_config, retention_config):
    """Test enhanced InfluxDB storage."""
    # Initialize storage
    storage = EnhancedInfluxDBStorage(
        url='http://localhost:8086',
        token='test-token',
        org='test-org',
        bucket='test-bucket',
        partition_config=partition_config,
        index_config=index_config,
        retention_config=retention_config
    )
    
    # Store data
    await storage.store(test_data)
    
    # Query data
    query = 'from(bucket:"test-bucket") |> range(start: -1h)'
    results = await storage.query(query)
    assert len(results) > 0

@pytest.mark.asyncio
async def test_partitioning_strategies(test_data):
    """Test different partitioning strategies."""
    # Time-based partitioning
    time_config = PartitionConfig(
        strategy=PartitionStrategy.TIME,
        field='timestamp',
        interval='hour'
    )
    
    # Symbol-based partitioning
    symbol_config = PartitionConfig(
        strategy=PartitionStrategy.SYMBOL,
        field='symbol'
    )
    
    # Hash-based partitioning
    hash_config = PartitionConfig(
        strategy=PartitionStrategy.HASH,
        field='symbol',
        num_partitions=5
    )
    
    # Composite partitioning
    composite_config = PartitionConfig(
        strategy=PartitionStrategy.COMPOSITE,
        field='timestamp',
        composite_fields=['symbol', 'timestamp']
    )
    
    # Test each strategy
    for config in [time_config, symbol_config, hash_config, composite_config]:
        storage = EnhancedParquetStorage(
            storage_path='./test_storage',
            partition_config=config
        )
        
        await storage.store(test_data)
        results = await storage.query()
        assert len(results) == len(test_data)
        
        # Cleanup
        import shutil
        shutil.rmtree('./test_storage')

@pytest.mark.asyncio
async def test_indexing_strategies(test_data):
    """Test different indexing strategies."""
    # B-tree index
    btree_config = IndexConfig(
        type=IndexType.BTREE,
        fields=['symbol', 'price']
    )
    
    # Hash index
    hash_config = IndexConfig(
        type=IndexType.HASH,
        fields=['symbol'],
        unique=True
    )
    
    # Test each strategy
    for config in [btree_config, hash_config]:
        storage = EnhancedParquetStorage(
            storage_path='./test_storage',
            index_config=config
        )
        
        await storage.store(test_data)
        results = await storage.query()
        assert len(results) == len(test_data)
        
        # Cleanup
        import shutil
        shutil.rmtree('./test_storage')

@pytest.mark.asyncio
async def test_retention_policies(test_data):
    """Test retention policies."""
    # Age-based retention
    age_config = RetentionConfig(
        max_age=timedelta(days=1),
        compression=True
    )
    
    # Size-based retention
    size_config = RetentionConfig(
        max_size=1024,  # 1KB
        compression=True
    )
    
    # Archive-based retention
    archive_config = RetentionConfig(
        max_age=timedelta(days=1),
        archive=True,
        archive_path='./archive'
    )
    
    # Test each policy
    for config in [age_config, size_config, archive_config]:
        storage = EnhancedParquetStorage(
            storage_path='./test_storage',
            retention_config=config
        )
        
        await storage.store(test_data)
        results = await storage.query()
        assert len(results) > 0
        
        # Cleanup
        import shutil
        shutil.rmtree('./test_storage')
        if os.path.exists('./archive'):
            shutil.rmtree('./archive')

@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling."""
    # Test invalid partition strategy
    with pytest.raises(ValueError):
        storage = EnhancedParquetStorage(
            storage_path='./test_storage',
            partition_config=PartitionConfig(
                strategy='invalid',
                field='timestamp'
            )
        )
    
    # Test invalid index type
    with pytest.raises(ValueError):
        storage = EnhancedParquetStorage(
            storage_path='./test_storage',
            index_config=IndexConfig(
                type='invalid',
                fields=['symbol']
            )
        )
    
    # Test invalid data
    storage = EnhancedParquetStorage(storage_path='./test_storage')
    with pytest.raises(Exception):
        await storage.store([{'invalid': 'data'}])
    
    # Cleanup
    import shutil
    if os.path.exists('./test_storage'):
        shutil.rmtree('./test_storage') 