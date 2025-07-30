"""Utility functions for the FeatureStore implementation."""

import json
import logging
import shutil
import dill
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Type, TypeVar, Generic, Callable
from datetime import datetime, timedelta
from enum import Enum, auto
from pydantic import BaseModel, Field, validator
from cachetools import LRUCache, TTLCache
from concurrent.futures import ThreadPoolExecutor
from threading import RLock
import tempfile
import io
import h5py
import pickle
import fnmatch
from abc import ABC, abstractmethod

# Configure logging
logger = logging.getLogger(__name__)

class FeatureExporter:
    """Handles exporting feature data to various formats."""
    
    @staticmethod
    def export_feature(
        data: Any,
        output_path: Union[str, Path],
        output_format: str = 'parquet',
        **kwargs
    ) -> str:
        """Export data to the specified format.
        
        Args:
            data: Data to export
            output_path: Path to save the exported file
            output_format: Output format ('parquet', 'csv', 'json', 'pickle', 'hdf5')
            **kwargs: Additional arguments for the export function
            
        Returns:
            str: Path to the exported file
        """
        output_path = Path(output_path)
        output_format = output_format.lower()
        
        if output_format == 'parquet':
            if not isinstance(data, pd.DataFrame):
                if isinstance(data, (np.ndarray, list, dict)):
                    data = pd.DataFrame(data)
                else:
                    data = pd.DataFrame({'value': [data]})
            data.to_parquet(output_path, **kwargs)
            
        elif output_format == 'csv':
            if not isinstance(data, pd.DataFrame):
                if isinstance(data, (np.ndarray, list, dict)):
                    data = pd.DataFrame(data)
                else:
                    data = pd.DataFrame({'value': [data]})
            data.to_csv(output_path, **kwargs)
            
        elif output_format == 'json':
            if isinstance(data, (pd.DataFrame, pd.Series)):
                data = data.to_dict(orient='records' if len(data.shape) > 1 else 'list')
            elif isinstance(data, np.ndarray):
                data = data.tolist()
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, **kwargs)
                
        elif output_format == 'pickle':
            with open(output_path, 'wb') as f:
                pickle.dump(data, f, **kwargs)
                
        elif output_format == 'hdf5':
            with h5py.File(output_path, 'w') as f:
                if isinstance(data, pd.DataFrame):
                    data.to_hdf(output_path, key='data', **kwargs)
                else:
                    f.create_dataset('data', data=np.array(data), **kwargs)
        else:
            raise ValueError(f"Unsupported export format: {output_format}")
            
        return str(output_path.absolute())


class FeatureImporter:
    """Handles importing feature data from various formats."""
    
    @staticmethod
    def import_data(
        data: Union[str, Path, bytes, Dict[str, Any], pd.DataFrame, np.ndarray],
        input_format: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Import data from the specified format.
        
        Args:
            data: Input data (file path, bytes, or data object)
            input_format: Input format ('parquet', 'csv', 'json', 'pickle', 'hdf5')
                         If None, will try to infer from file extension or data type
            **kwargs: Additional arguments for the import function
            
        Returns:
            Any: The imported data
        """
        # Determine input format if not specified
        if input_format is None:
            if isinstance(data, (str, Path)):
                input_format = str(data).split('.')[-1].lower()
            elif isinstance(data, bytes):
                input_format = 'pickle'  # Default for bytes
            elif isinstance(data, pd.DataFrame):
                return data  # Already a DataFrame
            elif isinstance(data, np.ndarray):
                return data  # Already a numpy array
            elif isinstance(data, (dict, list)):
                return data  # Already a dict/list
        else:
            input_format = input_format.lower()
        
        # Load data based on format
        if input_format in ('parquet', 'pq'):
            if isinstance(data, (str, Path)):
                return pd.read_parquet(data, **kwargs)
            elif isinstance(data, bytes):
                return pd.read_parquet(io.BytesIO(data), **kwargs)
                
        elif input_format == 'csv':
            if isinstance(data, (str, Path)):
                return pd.read_csv(data, **kwargs)
            elif isinstance(data, bytes):
                return pd.read_csv(io.BytesIO(data), **kwargs)
                
        elif input_format == 'json':
            if isinstance(data, (str, Path)):
                with open(data, 'r', encoding='utf-8') as f:
                    return json.load(f, **kwargs)
            elif isinstance(data, bytes):
                return json.loads(data.decode('utf-8'), **kwargs)
                
        elif input_format in ('pickle', 'pkl'):
            if isinstance(data, (str, Path)):
                with open(data, 'rb') as f:
                    return pickle.load(f, **kwargs)
            elif isinstance(data, bytes):
                return pickle.loads(data, **kwargs)
                
        elif input_format in ('hdf5', 'h5'):
            if isinstance(data, (str, Path)):
                with h5py.File(data, 'r') as f:
                    if 'data' in f:
                        return np.array(f['data'])
                    return {k: np.array(v) for k, v in f.items()}
            elif isinstance(data, bytes):
                with h5py.File(io.BytesIO(data), 'r') as f:
                    if 'data' in f:
                        return np.array(f['data'])
                    return {k: np.array(v) for k, v in f.items()}
        
        raise ValueError(f"Unsupported import format: {input_format}")


class FeatureBackup:
    """Handles backup and restore of feature stores."""
    
    @staticmethod
    def create_backup(
        feature_store: 'FeatureStore',
        output_path: Union[str, Path],
        include_data: bool = True
    ) -> bool:
        """Create a backup of a feature store.
        
        Args:
            feature_store: The feature store to back up
            output_path: Path to save the backup
            include_data: Whether to include feature data in the backup
            
        Returns:
            bool: True if backup was successful
        """
        try:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save metadata
            metadata_dir = output_path / 'metadata'
            metadata_dir.mkdir(exist_ok=True)
            
            for feature_name in feature_store.list_features():
                try:
                    metadata = feature_store.get_feature_metadata(feature_name, use_cache=False)
                    if metadata:
                        with open(metadata_dir / f"{feature_name}.json", 'w', encoding='utf-8') as f:
                            json.dump(metadata.to_dict(), f, indent=2, default=str)
                except Exception as e:
                    logger.error(f"Error backing up metadata for {feature_name}: {e}")
            
            # Save data if requested
            if include_data:
                data_dir = output_path / 'data'
                data_dir.mkdir(exist_ok=True)
                
                for feature_name in feature_store.list_features():
                    try:
                        versions = feature_store.list_versions(feature_name)
                        for version_info in versions:
                            version = version_info['version']
                            version_dir = data_dir / feature_name / version
                            version_dir.mkdir(parents=True, exist_ok=True)
                            
                            # Export data
                            data = feature_store.get_feature_data(feature_name, version=version, use_cache=False)
                            if data is not None:
                                if isinstance(data, pd.DataFrame):
                                    data.to_parquet(version_dir / 'data.parquet')
                                elif isinstance(data, np.ndarray):
                                    np.save(version_dir / 'data.npy', data)
                                elif isinstance(data, (dict, list)):
                                    with open(version_dir / 'data.json', 'w', encoding='utf-8') as f:
                                        json.dump(data, f, default=str)
                    except Exception as e:
                        logger.error(f"Error backing up data for {feature_name}: {e}")
            
            # Save backup metadata
            backup_meta = {
                'timestamp': datetime.utcnow().isoformat(),
                'feature_count': len(feature_store.list_features()),
                'include_data': include_data,
                'version': '1.0.0'
            }
            
            with open(output_path / 'backup_meta.json', 'w', encoding='utf-8') as f:
                json.dump(backup_meta, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}", exc_info=True)
            return False
    
    @classmethod
    def restore_backup(
        cls,
        backup_path: Union[str, Path],
        feature_store: 'FeatureStore',
        include_data: bool = True
    ) -> bool:
        """Restore a feature store from a backup.
        
        Args:
            backup_path: Path to the backup directory
            feature_store: The feature store to restore to
            include_data: Whether to restore feature data
            
        Returns:
            bool: True if restore was successful
        """
        try:
            backup_path = Path(backup_path)
            
            # Read backup metadata
            with open(backup_path / 'backup_meta.json', 'r', encoding='utf-8') as f:
                backup_meta = json.load(f)
            
            # Restore metadata
            metadata_dir = backup_path / 'metadata'
            if metadata_dir.exists():
                for meta_file in metadata_dir.glob('*.json'):
                    try:
                        with open(meta_file, 'r', encoding='utf-8') as f:
                            meta_dict = json.load(f)
                        
                        # Create feature with metadata
                        feature_store.register_feature(meta_dict)
                    except Exception as e:
                        logger.error(f"Error restoring metadata from {meta_file}: {e}")
            
            # Restore data if available and requested
            data_dir = backup_path / 'data'
            if include_data and data_dir.exists() and backup_meta.get('include_data', True):
                for feature_dir in data_dir.iterdir():
                    if not feature_dir.is_dir():
                        continue
                        
                    feature_name = feature_dir.name
                    for version_dir in feature_dir.iterdir():
                        if not version_dir.is_dir():
                            continue
                            
                        version = version_dir.name
                        try:
                            # Find the data file
                            data_file = None
                            for ext in ['.parquet', '.npy', '.json']:
                                if (version_dir / f'data{ext}').exists():
                                    data_file = version_dir / f'data{ext}'
                                    break
                                    
                            if data_file is None:
                                logger.warning(f"No data file found for {feature_name} v{version}")
                                continue
                            
                            # Import data based on file extension
                            data = FeatureImporter.import_data(
                                data_file,
                                input_format=data_file.suffix[1:]
                            )
                            
                            # Save the feature data
                            feature_store.save_feature_data(
                                name=feature_name,
                                data=data,
                                version=version
                            )
                            
                        except Exception as e:
                            logger.error(f"Error restoring data for {feature_name} v{version}: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error restoring from backup: {e}", exc_info=True)
            return False
