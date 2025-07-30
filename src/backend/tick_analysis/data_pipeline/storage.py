"""
Enhanced Storage Module

This module provides comprehensive data storage capabilities including
partitioning, indexing, retention policies, and compression.
"""

import logging
import asyncio
import json
import os
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import aiofiles
import pandas as pd
import numpy as np
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
from enum import Enum

logger = logging.getLogger(__name__)

class StorageType(Enum):
    """Types of storage backends."""
    INFLUXDB = "influxdb"
    PARQUET = "parquet"

class PartitionStrategy(Enum):
    """Strategies for data partitioning."""
    TIME = "time"
    SYMBOL = "symbol"
    HASH = "hash"
    COMPOSITE = "composite"

class IndexType(Enum):
    """Types of indexes."""
    BTREE = "btree"
    HASH = "hash"
    SPATIAL = "spatial"

@dataclass
class PartitionConfig:
    """Configuration for data partitioning."""
    strategy: PartitionStrategy
    field: str
    interval: Optional[str] = None  # For time-based partitioning
    num_partitions: Optional[int] = None  # For hash-based partitioning
    composite_fields: Optional[List[str]] = None  # For composite partitioning

@dataclass
class IndexConfig:
    """Configuration for indexes."""
    type: IndexType
    fields: List[str]
    unique: bool = False
    sparse: bool = False

@dataclass
class RetentionConfig:
    """Configuration for data retention."""
    max_age: timedelta
    max_size: Optional[int] = None  # In bytes
    compression: bool = True
    archive: bool = False
    archive_path: Optional[str] = None

class DataPartitioner:
    """Handles data partitioning strategies."""
    
    def __init__(self, config: PartitionConfig):
        """
        Initialize the partitioner.
        
        Args:
            config: Partitioning configuration
        """
        self.config = config
    
    def get_partition_key(self, data: Dict[str, Any]) -> str:
        """
        Get partition key for data.
        
        Args:
            data: Data to partition
            
        Returns:
            Partition key
        """
        if self.config.strategy == PartitionStrategy.TIME:
            timestamp = pd.to_datetime(data[self.config.field])
            if self.config.interval == 'day':
                return timestamp.strftime('%Y-%m-%d')
            elif self.config.interval == 'hour':
                return timestamp.strftime('%Y-%m-%d-%H')
            elif self.config.interval == 'minute':
                return timestamp.strftime('%Y-%m-%d-%H-%M')
            else:
                return timestamp.strftime('%Y-%m-%d')
                
        elif self.config.strategy == PartitionStrategy.SYMBOL:
            return str(data[self.config.field])
            
        elif self.config.strategy == PartitionStrategy.HASH:
            value = str(data[self.config.field])
            return str(hash(value) % self.config.num_partitions)
            
        elif self.config.strategy == PartitionStrategy.COMPOSITE:
            values = [str(data[field]) for field in self.config.composite_fields]
            return '-'.join(values)
            
        else:
            raise ValueError(f"Unsupported partition strategy: {self.config.strategy}")
    
    def partition_data(self, data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Partition data into groups.
        
        Args:
            data: List of data records
            
        Returns:
            Dictionary mapping partition keys to data records
        """
        partitions = {}
        
        for record in data:
            key = self.get_partition_key(record)
            if key not in partitions:
                partitions[key] = []
            partitions[key].append(record)
        
        return partitions

class IndexManager:
    """Manages data indexes."""
    
    def __init__(self, config: IndexConfig):
        """
        Initialize the index manager.
        
        Args:
            config: Index configuration
        """
        self.config = config
        self.indexes = {}
    
    def create_index(self, data: List[Dict[str, Any]]) -> None:
        """
        Create index for data.
        
        Args:
            data: Data to index
        """
        if self.config.type == IndexType.BTREE:
            self._create_btree_index(data)
        elif self.config.type == IndexType.HASH:
            self._create_hash_index(data)
        elif self.config.type == IndexType.SPATIAL:
            self._create_spatial_index(data)
        else:
            raise ValueError(f"Unsupported index type: {self.config.type}")
    
    def _create_btree_index(self, data: List[Dict[str, Any]]) -> None:
        """Create B-tree index."""
        for field in self.config.fields:
            values = [(i, record[field]) for i, record in enumerate(data)]
            values.sort(key=lambda x: x[1])
            self.indexes[field] = values
    
    def _create_hash_index(self, data: List[Dict[str, Any]]) -> None:
        """Create hash index."""
        for field in self.config.fields:
            index = {}
            for i, record in enumerate(data):
                value = record[field]
                if value not in index:
                    index[value] = []
                index[value].append(i)
            self.indexes[field] = index
    
    def _create_spatial_index(self, data: List[Dict[str, Any]]) -> None:
        """Create spatial index."""
        # Implement spatial indexing (e.g., R-tree) if needed
        pass
    
    def query_index(self, field: str, value: Any) -> List[int]:
        """
        Query index for records.
        
        Args:
            field: Field to query
            value: Value to search for
            
        Returns:
            List of record indices
        """
        if field not in self.indexes:
            return []
            
        if self.config.type == IndexType.BTREE:
            return self._query_btree_index(field, value)
        elif self.config.type == IndexType.HASH:
            return self._query_hash_index(field, value)
        elif self.config.type == IndexType.SPATIAL:
            return self._query_spatial_index(field, value)
        else:
            return []
    
    def _query_btree_index(self, field: str, value: Any) -> List[int]:
        """Query B-tree index."""
        indices = []
        for i, v in self.indexes[field]:
            if v == value:
                indices.append(i)
            elif v > value:
                break
        return indices
    
    def _query_hash_index(self, field: str, value: Any) -> List[int]:
        """Query hash index."""
        return self.indexes[field].get(value, [])
    
    def _query_spatial_index(self, field: str, value: Any) -> List[int]:
        """Query spatial index."""
        # Implement spatial query if needed
        return []

class RetentionManager:
    """Manages data retention policies."""
    
    def __init__(self, config: RetentionConfig):
        """
        Initialize the retention manager.
        
        Args:
            config: Retention configuration
        """
        self.config = config
    
    async def apply_policies(self, data: List[Dict[str, Any]], storage_path: str) -> None:
        """
        Apply retention policies to data.
        
        Args:
            data: Data to process
            storage_path: Path to storage
        """
        # Check age
        if self.config.max_age:
            data = self._filter_by_age(data)
        
        # Check size
        if self.config.max_size:
            data = self._filter_by_size(data, storage_path)
        
        # Apply compression
        if self.config.compression:
            await self._compress_data(data, storage_path)
        
        # Archive if needed
        if self.config.archive and self.config.archive_path:
            await self._archive_data(data, storage_path)
    
    def _filter_by_age(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter data by age."""
        cutoff = datetime.utcnow() - self.config.max_age
        return [record for record in data 
                if pd.to_datetime(record['timestamp']) > cutoff]
    
    def _filter_by_size(self, data: List[Dict[str, Any]], storage_path: str) -> List[Dict[str, Any]]:
        """Filter data by size."""
        current_size = sum(os.path.getsize(os.path.join(storage_path, f))
                         for f in os.listdir(storage_path))
        
        if current_size <= self.config.max_size:
            return data
            
        # Remove oldest data until under size limit
        sorted_data = sorted(data, key=lambda x: x['timestamp'])
        while current_size > self.config.max_size and sorted_data:
            sorted_data.pop(0)
            # Recalculate size (simplified)
            current_size = len(json.dumps(sorted_data))
        
        return sorted_data
    
    async def _compress_data(self, data: List[Dict[str, Any]], storage_path: str) -> None:
        """Compress data."""
        # Convert to Parquet with compression
        df = pd.DataFrame(data)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, os.path.join(storage_path, 'compressed.parquet'),
                      compression='snappy')
    
    async def _archive_data(self, data: List[Dict[str, Any]], storage_path: str) -> None:
        """Archive data."""
        os.makedirs(self.config.archive_path, exist_ok=True)
        
        # Move data to archive
        archive_file = os.path.join(self.config.archive_path,
                                   f"archive_{datetime.utcnow().strftime('%Y%m%d')}.parquet")
        
        df = pd.DataFrame(data)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, archive_file)

class EnhancedInfluxDBStorage:
    """Enhanced InfluxDB storage with partitioning and indexing."""
    
    def __init__(self,
                 url: str,
                 token: str,
                 org: str,
                 bucket: str,
                 partition_config: Optional[PartitionConfig] = None,
                 index_config: Optional[IndexConfig] = None,
                 retention_config: Optional[RetentionConfig] = None):
        """
        Initialize InfluxDB storage.
        
        Args:
            url: InfluxDB URL
            token: Authentication token
            org: Organization name
            bucket: Bucket name
            partition_config: Optional partitioning configuration
            index_config: Optional indexing configuration
            retention_config: Optional retention configuration
        """
        self.client = InfluxDBClient(url=url, token=token, org=org)
        self.bucket = bucket
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        
        self.partitioner = DataPartitioner(partition_config) if partition_config else None
        self.indexer = IndexManager(index_config) if index_config else None
        self.retention_manager = RetentionManager(retention_config) if retention_config else None
    
    async def store(self, data: List[Dict[str, Any]]) -> None:
        """
        Store data in InfluxDB.
        
        Args:
            data: Data to store
        """
        if self.partitioner:
            partitions = self.partitioner.partition_data(data)
        else:
            partitions = {'default': data}
        
        for partition_key, partition_data in partitions.items():
            points = []
            for record in partition_data:
                point = Point(partition_key)
                for key, value in record.items():
                    if isinstance(value, (int, float)):
                        point.field(key, value)
                    elif isinstance(value, str):
                        point.tag(key, value)
                    elif isinstance(value, datetime):
                        point.time(value)
                points.append(point)
            
            self.write_api.write(bucket=self.bucket, record=points)
    
    async def query(self, query: str) -> List[Dict[str, Any]]:
        """
        Query data from InfluxDB.
        
        Args:
            query: Flux query string
            
        Returns:
            Query results
        """
        result = self.client.query_api().query(query, org=self.client.org)
        return [record.values for table in result for record in table]

class EnhancedParquetStorage:
    """Enhanced Parquet storage with partitioning and indexing."""
    
    def __init__(self,
                 storage_path: str,
                 partition_config: Optional[PartitionConfig] = None,
                 index_config: Optional[IndexConfig] = None,
                 retention_config: Optional[RetentionConfig] = None):
        """
        Initialize Parquet storage.
        
        Args:
            storage_path: Path to store data
            partition_config: Optional partitioning configuration
            index_config: Optional indexing configuration
            retention_config: Optional retention configuration
        """
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        
        self.partitioner = DataPartitioner(partition_config) if partition_config else None
        self.indexer = IndexManager(index_config) if index_config else None
        self.retention_manager = RetentionManager(retention_config) if retention_config else None
    
    async def store(self, data: List[Dict[str, Any]]) -> None:
        """
        Store data in Parquet format.
        
        Args:
            data: Data to store
        """
        if self.partitioner:
            partitions = self.partitioner.partition_data(data)
        else:
            partitions = {'default': data}
        
        for partition_key, partition_data in partitions.items():
            # Create partition directory
            partition_path = os.path.join(self.storage_path, partition_key)
            os.makedirs(partition_path, exist_ok=True)
            
            # Convert to DataFrame and write to Parquet
            df = pd.DataFrame(partition_data)
            table = pa.Table.from_pandas(df)
            
            # Add partitioning metadata
            metadata = {
                'partition_key': partition_key,
                'timestamp': datetime.utcnow().isoformat()
            }
            table = table.replace_schema_metadata(metadata)
            
            # Write to Parquet
            file_path = os.path.join(partition_path, f"data_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.parquet")
            pq.write_table(table, file_path)
            
            # Create indexes if configured
            if self.indexer:
                self.indexer.create_index(partition_data)
            
            # Apply retention policies if configured
            if self.retention_manager:
                await self.retention_manager.apply_policies(partition_data, partition_path)
    
    async def query(self,
                   partition_key: Optional[str] = None,
                   filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Query data from Parquet files.
        
        Args:
            partition_key: Optional partition key to query
            filters: Optional filters to apply
            
        Returns:
            Query results
        """
        results = []
        
        # Determine partitions to query
        if partition_key:
            partitions = [partition_key]
        else:
            partitions = os.listdir(self.storage_path)
        
        for partition in partitions:
            partition_path = os.path.join(self.storage_path, partition)
            if not os.path.isdir(partition_path):
                continue
            
            # Read all Parquet files in partition
            for file_name in os.listdir(partition_path):
                if not file_name.endswith('.parquet'):
                    continue
                
                file_path = os.path.join(partition_path, file_name)
                table = pq.read_table(file_path)
                df = table.to_pandas()
                
                # Apply filters
                if filters:
                    for field, value in filters.items():
                        df = df[df[field] == value]
                
                results.extend(df.to_dict('records'))
        
        return results
