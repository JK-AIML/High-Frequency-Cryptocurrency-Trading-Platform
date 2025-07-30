from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict, fields
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Union, Tuple, Callable, Type, TypeVar, Generic
import logging
import json
import shutil
import dill
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import io
import h5py
import pickle
import fnmatch
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from threading import RLock
from cachetools import TTLCache, LRUCache
from pydantic import BaseModel, Field, validator

# Import utility classes
from .feature_store_utils import FeatureExporter, FeatureImporter, FeatureBackup

# Configure logging
logger = logging.getLogger(__name__)

# Type variables for generic feature types
T = TypeVar('T')
F = TypeVar('F', bound='BaseFeature')

# Constants
DEFAULT_CACHE_SIZE = 10000
DEFAULT_CACHE_TTL = 300  # 5 minutes
MAX_FEATURE_NAME_LENGTH = 255
VALID_FEATURE_NAME_PATTERN = r'^[a-zA-Z0-9_\-]+$'
DEFAULT_PARTITION_COLUMNS = ['year', 'month', 'day']
DEFAULT_STORAGE_FORMAT = 'parquet'
DEFAULT_COMPRESSION = 'snappy'

# Feature Store Events
class FeatureStoreEventType(str, Enum):
    FEATURE_CREATED = "feature_created"
    FEATURE_UPDATED = "feature_updated"
    FEATURE_DELETED = "feature_deleted"
    FEATURE_ACCESSED = "feature_accessed"
    FEATURE_VERSION_CREATED = "feature_version_created"
    FEATURE_VERSION_DELETED = "feature_version_deleted"
    FEATURE_STATISTICS_UPDATED = "feature_statistics_updated"
    FEATURE_LINEAGE_UPDATED = "feature_lineage_updated"
    FEATURE_STORE_INITIALIZED = "feature_store_initialized"
    FEATURE_STORE_BACKUP_CREATED = "feature_store_backup_created"
    FEATURE_STORE_RESTORED = "feature_store_restored"

class FeatureType(Enum):
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    EMBEDDING = "embedding"
    TIME_SERIES = "time_series"
    TRANSFORMED = "transformed"

class FeatureStatus(Enum):
    EXPERIMENTAL = "experimental"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"

@dataclass
class FeatureVersion(BaseModel):
    """Represents a version of a feature."""
    version: str = Field(..., description="Semantic version string (e.g., '1.0.0')")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When this version was created")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="When this version was last updated")
    description: str = Field("", description="Description of changes in this version")
    tags: List[str] = Field(default_factory=list, description="Tags for this version")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    is_production: bool = Field(False, description="Whether this is a production version")
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat(),
        }
    
    def __str__(self) -> str:
        return f"v{self.version} ({'prod' if self.is_production else 'dev'})"


class FeatureStatistics(BaseModel):
    """Statistical information about a feature's data."""
    count: Optional[int] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    p25: Optional[float] = None
    p50: Optional[float] = None
    p75: Optional[float] = None
    p95: Optional[float] = None
    p99: Optional[float] = None
    null_count: Optional[int] = None
    distinct_count: Optional[int] = None
    data_type: Optional[str] = None
    last_updated: Optional[datetime] = None
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat(),
        }


class FeatureLineage(BaseModel):
    """Lineage information for a feature."""
    source_features: List[str] = Field(default_factory=list, description="Features this feature depends on")
    transformations: List[Dict[str, Any]] = Field(default_factory=list, description="Transformations applied to create this feature")
    created_by: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class FeatureMetadata(BaseModel):
    """Metadata for a feature with versioning support."""
    name: str = Field(..., max_length=MAX_FEATURE_NAME_LENGTH, regex=VALID_FEATURE_NAME_PATTERN)
    namespace: str = "default"
    description: str = ""
    feature_type: FeatureType = Field(..., description="Type of the feature")
    data_type: str = Field(..., description="Data type of the feature values")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    updated_by: Optional[str] = None
    
    # Versioning
    version: str = "1.0.0"
    versions: Dict[str, FeatureVersion] = Field(default_factory=dict)
    current_version: Optional[str] = None
    
    # Schema and validation
    schema: Dict[str, Any] = Field(default_factory=dict, description="Schema definition for the feature")
    validation_rules: Dict[str, Any] = Field(default_factory=dict, description="Validation rules")
    
    # Statistics and monitoring
    statistics: Dict[str, FeatureStatistics] = Field(default_factory=dict, description="Statistical information by version")
    monitoring_config: Dict[str, Any] = Field(default_factory=dict, description="Monitoring configuration")
    
    # Organization and access control
    owner: str = ""
    team: str = ""
    tags: List[str] = Field(default_factory=list)
    access_control: Dict[str, List[str]] = Field(default_factory=dict, description="Access control list")
    
    # Lineage and dependencies
    lineage: Optional[FeatureLineage] = None
    dependencies: List[str] = Field(default_factory=list, description="Dependent features")
    
    # Storage configuration
    storage_format: str = DEFAULT_STORAGE_FORMAT
    compression: str = DEFAULT_COMPRESSION
    partition_columns: List[str] = Field(default_factory=lambda: DEFAULT_PARTITION_COLUMNS.copy())
    
    # Status and lifecycle
    status: FeatureStatus = FeatureStatus.EXPERIMENTAL
    is_public: bool = False
    is_cached: bool = True
    ttl_days: Optional[int] = None
    
    # Custom metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata")
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat(),
            FeatureType: lambda ft: ft.value,
            FeatureStatus: lambda fs: fs.value,
        }
        validate_assignment = True
        extra = "forbid"
    
    def __init__(self, **data):
        super().__init__(**data)
        # Initialize versions if not present
        if not self.versions:
            self.versions = {
                self.version: FeatureVersion(
                    version=self.version,
                    is_production=self.status == FeatureStatus.PRODUCTION,
                    description="Initial version"
                )
            }
        if self.current_version is None:
            self.current_version = self.version
    
    @validator('name')
    def validate_name(cls, v):
        if not re.match(VALID_FEATURE_NAME_PATTERN, v):
            raise ValueError(f"Feature name '{v}' contains invalid characters. Only alphanumeric, underscore, and hyphen are allowed.")
        if len(v) > MAX_FEATURE_NAME_LENGTH:
            raise ValueError(f"Feature name '{v}' exceeds maximum length of {MAX_FEATURE_NAME_LENGTH} characters.")
        return v
    
    @validator('version')
    def validate_version(cls, v):
        try:
            # Simple semver validation
            parts = [int(x) for x in v.split('.')]
            if len(parts) != 3 or any(p < 0 for p in parts):
                raise ValueError("Version must be in format 'X.Y.Z' where X, Y, Z are non-negative integers")
            return v
        except (ValueError, AttributeError):
            raise ValueError("Version must be in format 'X.Y.Z' where X, Y, Z are non-negative integers")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper serialization."""
        data = self.dict(exclude_unset=True)
        data['feature_type'] = self.feature_type.value
        data['status'] = self.status.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureMetadata':
        """Create from dictionary with proper deserialization."""
        if 'feature_type' in data:
            data['feature_type'] = FeatureType(data['feature_type'])
        if 'status' in data:
            data['status'] = FeatureStatus(data['status'])
        
        # Handle versions
        versions = data.pop('versions', {})
        if isinstance(versions, list):
            versions = {ver['version']: ver for ver in versions if 'version' in ver}
        
        # Create instance
        instance = cls(**data)
        
        # Set versions
        instance.versions = {
            ver: FeatureVersion.parse_obj(ver_data) 
            for ver, ver_data in versions.items()
        }
        
        return instance
    
    def create_new_version(self, version: str, description: str = "", is_production: bool = False) -> 'FeatureMetadata':
        """Create a new version of this feature."""
        if version in self.versions:
            raise ValueError(f"Version {version} already exists")
        
        # Create new version
        new_version = FeatureVersion(
            version=version,
            description=description,
            is_production=is_production,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        # Update metadata
        self.versions[version] = new_version
        self.version = version
        self.current_version = version
        self.updated_at = datetime.utcnow()
        
        if is_production:
            self.status = FeatureStatus.PRODUCTION
        
        return self
    
    def get_version(self, version: Optional[str] = None) -> Optional[FeatureVersion]:
        """Get a specific version or the current version if None."""
        version = version or self.current_version or self.version
        return self.versions.get(version)
    
    def promote_to_production(self, version: Optional[str] = None) -> None:
        """Promote a version to production."""
        version = version or self.current_version or self.version
        if version not in self.versions:
            raise ValueError(f"Version {version} not found")
        
        # Demote current production version if exists
        for ver in self.versions.values():
            if ver.is_production and ver.version != version:
                ver.is_production = False
        
        # Promote the specified version
        self.versions[version].is_production = True
        self.status = FeatureStatus.PRODUCTION
        self.current_version = version
        self.version = version
        self.updated_at = datetime.utcnow()
    
    def validate_data(self, data: Any) -> bool:
        """Validate data against the feature's schema and validation rules."""
        # Basic type checking
        if self.data_type == 'numeric' and not isinstance(data, (int, float)):
            return False
        elif self.data_type == 'categorical' and not isinstance(data, (str, int)):
            return False
        # Add more validations as needed
        return True
    
    def update_statistics(self, data: Any, version: Optional[str] = None) -> None:
        """Update statistics for this feature."""
        version = version or self.current_version or self.version
        if version not in self.versions:
            raise ValueError(f"Version {version} not found")
        
        # Calculate statistics
        stats = self.statistics.get(version, FeatureStatistics())
        
        # Update statistics based on data
        # This is a simplified example - in practice, you'd use a proper statistics library
        if isinstance(data, pd.Series):
            stats.count = len(data)
            if pd.api.types.is_numeric_dtype(data):
                stats.mean = float(data.mean())
                stats.std = float(data.std())
                stats.min = float(data.min())
                stats.max = float(data.max())
                stats.p25 = float(data.quantile(0.25))
                stats.p50 = float(data.median())
                stats.p75 = float(data.quantile(0.75))
                stats.p95 = float(data.quantile(0.95))
                stats.p99 = float(data.quantile(0.99))
            stats.null_count = int(data.isnull().sum())
            stats.distinct_count = len(data.unique())
            stats.data_type = str(data.dtype)
        
        stats.last_updated = datetime.utcnow()
        self.statistics[version] = stats

class StorageBackend(ABC):
    """Abstract base class for storage backends."""
    
    @abstractmethod
    def save_metadata(self, feature_name: str, metadata: Dict[str, Any]) -> bool:
        """Save feature metadata."""
        pass
    
    @abstractmethod
    def load_metadata(self, feature_name: str) -> Optional[Dict[str, Any]]:
        """Load feature metadata."""
        pass
    
    @abstractmethod
    def delete_metadata(self, feature_name: str) -> bool:
        """Delete feature metadata."""
        pass
    
    @abstractmethod
    def save_data(self, feature_name: str, version: str, data: Any, **kwargs) -> bool:
        """Save feature data."""
        pass
    
    @abstractmethod
    def load_data(self, feature_name: str, version: str, **kwargs) -> Any:
        """Load feature data."""
        pass
    
    @abstractmethod
    def delete_data(self, feature_name: str, version: str) -> bool:
        """Delete feature data."""
        pass
    
    @abstractmethod
    def list_features(self) -> List[str]:
        """List all features."""
        pass
    
    @abstractmethod
    def list_versions(self, feature_name: str) -> List[str]:
        """List all versions of a feature."""
        pass


class FilesystemStorageBackend(StorageBackend):
    """Filesystem-based storage backend."""
    
    def __init__(self, base_path: Union[str, Path], **kwargs):
        self.base_path = Path(base_path)
        self.compression = kwargs.get('compression', 'snappy')
        self._create_directories()
    
    def _create_directories(self):
        """Create required directories."""
        (self.base_path / 'metadata').mkdir(parents=True, exist_ok=True)
        (self.base_path / 'data').mkdir(parents=True, exist_ok=True)
        (self.base_path / 'cache').mkdir(parents=True, exist_ok=True)
    
    def _get_metadata_path(self, feature_name: str) -> Path:
        """Get path for feature metadata."""
        return self.base_path / 'metadata' / f"{feature_name}.json"
    
    def _get_data_path(self, feature_name: str, version: str) -> Path:
        """Get base path for feature data."""
        return self.base_path / 'data' / feature_name / version
    
    def save_metadata(self, feature_name: str, metadata: Dict[str, Any]) -> bool:
        try:
            path = self._get_metadata_path(feature_name)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, default=str)
            return True
        except Exception as e:
            logger.error(f"Error saving metadata for {feature_name}: {e}")
            return False
    
    def load_metadata(self, feature_name: str) -> Optional[Dict[str, Any]]:
        try:
            path = self._get_metadata_path(feature_name)
            if not path.exists():
                return None
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading metadata for {feature_name}: {e}")
            return None
    
    def delete_metadata(self, feature_name: str) -> bool:
        try:
            path = self._get_metadata_path(feature_name)
            if path.exists():
                path.unlink()
            return True
        except Exception as e:
            logger.error(f"Error deleting metadata for {feature_name}: {e}")
            return False
    
    def save_data(self, feature_name: str, version: str, data: Any, **kwargs) -> bool:
        try:
            base_path = self._get_data_path(feature_name, version)
            base_path.mkdir(parents=True, exist_ok=True)
            
            if isinstance(data, pd.DataFrame):
                # Save as parquet by default
                path = base_path / 'data.parquet'
                data.to_parquet(path, compression=self.compression, **kwargs)
            elif isinstance(data, (np.ndarray, list, dict)):
                # Save as numpy array or JSON
                path = base_path / 'data.npz' if isinstance(data, np.ndarray) else base_path / 'data.json'
                if isinstance(data, np.ndarray):
                    np.savez_compressed(path, data=data)
                else:
                    with open(path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, default=str)
            else:
                # Try to serialize with dill as last resort
                path = base_path / 'data.pkl'
                with open(path, 'wb') as f:
                    dill.dump(data, f)
            
            # Save metadata about the data
            meta = {
                'saved_at': datetime.utcnow().isoformat(),
                'format': path.suffix[1:],
                'size_bytes': path.stat().st_size if path.exists() else 0,
                'dtype': str(type(data)),
                'shape': getattr(data, 'shape', None),
                'columns': list(data.columns) if hasattr(data, 'columns') else None
            }
            with open(base_path / 'meta.json', 'w', encoding='utf-8') as f:
                json.dump(meta, f, indent=2)
                
            return True
            
        except Exception as e:
            logger.error(f"Error saving data for {feature_name} v{version}: {e}")
            return False
    
    def load_data(self, feature_name: str, version: str, **kwargs) -> Any:
        try:
            base_path = self._get_data_path(feature_name, version)
            
            # Check for different file formats
            for ext in ['.parquet', '.npz', '.json', '.pkl']:
                path = base_path / f'data{ext}'
                if path.exists():
                    if ext == '.parquet':
                        return pd.read_parquet(path, **kwargs)
                    elif ext == '.npz':
                        return np.load(path)['data']
                    elif ext == '.json':
                        with open(path, 'r', encoding='utf-8') as f:
                            return json.load(f)
                    else:  # .pkl
                        with open(path, 'rb') as f:
                            return dill.load(f)
            
            logger.warning(f"No data found for {feature_name} v{version}")
            return None
            
        except Exception as e:
            logger.error(f"Error loading data for {feature_name} v{version}: {e}")
            return None
    
    def delete_data(self, feature_name: str, version: str) -> bool:
        try:
            path = self._get_data_path(feature_name, version)
            if path.exists():
                shutil.rmtree(path)
            return True
        except Exception as e:
            logger.error(f"Error deleting data for {feature_name} v{version}: {e}")
            return False
    
    def list_features(self) -> List[str]:
        try:
            return [f.stem for f in (self.base_path / 'metadata').glob('*.json')]
        except Exception as e:
            logger.error(f"Error listing features: {e}")
            return []
    
    def list_versions(self, feature_name: str) -> List[str]:
        try:
            feature_path = self.base_path / 'data' / feature_name
            if not feature_path.exists():
                return []
            return [d.name for d in feature_path.iterdir() if d.is_dir()]
        except Exception as e:
            logger.error(f"Error listing versions for {feature_name}: {e}")
            return []


class FeatureStore:
    """
    Feature store implementation with versioning, metadata management, and online serving.
    
    Features:
    - Version control for features
    - Metadata management with schema validation
    - Online/offline storage
    - Point-in-time correctness
    - Batch and streaming support
    - Caching and performance optimizations
    - Access control and audit logging
    """
    
    def __init__(
        self, 
        base_path: Optional[Union[str, Path]] = None,
        storage_backend: Optional[Union[str, StorageBackend]] = None,
        cache_size: int = DEFAULT_CACHE_SIZE,
        cache_ttl: int = DEFAULT_CACHE_TTL,
        enable_online_serving: bool = False,
        online_serving_port: int = 8000,
        **kwargs
    ):
        """Initialize the feature store.
        
        Args:
            base_path: Base path for storage (for filesystem backend)
            storage_backend: Storage backend instance or name ('filesystem', 's3', 'gcs', etc.)
            cache_size: Maximum number of items to keep in cache
            cache_ttl: Time-to-live for cache items in seconds
            enable_online_serving: Whether to start the online serving API
            online_serving_port: Port for the online serving API
            **kwargs: Additional backend-specific arguments
        """
        # Initialize storage backend
        if storage_backend is None:
            if base_path is None:
                raise ValueError("Either base_path or storage_backend must be provided")
            self.storage = FilesystemStorageBackend(base_path, **kwargs)
        elif isinstance(storage_backend, str):
            if storage_backend.lower() == 'filesystem':
                self.storage = FilesystemStorageBackend(base_path or '.', **kwargs)
            # Add other backends (S3, GCS, etc.) as needed
            else:
                raise ValueError(f"Unsupported storage backend: {storage_backend}")
        else:
            self.storage = storage_backend
        
        # Initialize caches
        self._metadata_cache = LRUCache(maxsize=cache_size)
        self._data_cache = TTLCache(maxsize=cache_size, ttl=cache_ttl)
        self._stats_cache = TTLCache(maxsize=cache_size, ttl=cache_ttl // 2)
        self._cache_lock = RLock()
        
        # Online serving
        self.enable_online_serving = enable_online_serving
        self.online_serving_port = online_serving_port
        self._fastapi_app = None
        
        # Background tasks
        self._executor = ThreadPoolExecutor(
            max_workers=kwargs.get('max_workers', 4),
            thread_name_prefix='feature_store_worker'
        )
        
        # Load initial metadata
        self._load_metadata_cache()
        
        # Start online serving if enabled
        if self.enable_online_serving:
            self._init_online_serving()
    
    def _load_metadata_cache(self) -> None:
        """Load metadata from storage into memory cache."""
        with self._cache_lock:
            for feature_name in self.storage.list_features():
                try:
                    # Skip if already in cache
                    if feature_name in self._metadata_cache:
                        continue
                        
                    # Load from storage
                    meta_dict = self.storage.load_metadata(feature_name)
                    if not meta_dict:
                        continue
                        
                    # Convert to FeatureMetadata
                    metadata = FeatureMetadata.from_dict(meta_dict)
                    self._metadata_cache[feature_name] = metadata
                    
                except Exception as e:
                    logger.error(f"Error loading metadata for {feature_name}: {e}")
    
    def _save_metadata(self, metadata: FeatureMetadata) -> bool:
        """Save metadata to storage."""
        try:
            # Update timestamps
            now = datetime.utcnow()
            if not metadata.created_at:
                metadata.created_at = now
            metadata.updated_at = now
            
            # Save to storage
            success = self.storage.save_metadata(
                feature_name=metadata.name,
                metadata=metadata.to_dict()
            )
            
            if success:
                # Update cache
                with self._cache_lock:
                    self._metadata_cache[metadata.name] = metadata
                
                # Log the event
                self._log_event(
                    event_type=FeatureStoreEventType.FEATURE_UPDATED,
                    feature_name=metadata.name,
                    version=metadata.version,
                    metadata={'action': 'metadata_updated'}
                )
                
            return success
            
        except Exception as e:
            logger.error(f"Error saving metadata for {metadata.name}: {e}")
            return False
    
    def _log_event(self, event_type: FeatureStoreEventType, feature_name: str, **kwargs) -> None:
        """Log an event to the audit log."""
        event = {
            'event_type': event_type.value,
            'feature_name': feature_name,
            'timestamp': datetime.utcnow().isoformat(),
            **kwargs
        }
        logger.info(f"Feature Store Event: {json.dumps(event, default=str)}")
    
    def register_feature(
        self, 
        metadata: Union[FeatureMetadata, Dict[str, Any]],
        version: Optional[str] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
        is_production: bool = False,
        **kwargs
    ) -> bool:
        """Register or update a feature with versioning support.
        
        Args:
            metadata: Feature metadata as a FeatureMetadata instance or dict
            version: Optional version string (e.g., '1.0.0'). If None, auto-increments.
            description: Description of this version
            tags: Tags for this version
            is_production: Whether to mark this version as production
            **kwargs: Additional metadata fields
            
        Returns:
            bool: True if registration was successful
        """
        try:
            # Convert dict to FeatureMetadata if needed
            if isinstance(metadata, dict):
                metadata = FeatureMetadata(**{**metadata, **kwargs})
            
            # Check for existing feature
            existing = self.get_feature_metadata(metadata.name, use_cache=False)
            
            if existing:
                # Update existing feature
                if version and version in [v.version for v in existing.versions.values()]:
                    raise ValueError(f"Version {version} already exists for feature {metadata.name}")
                
                # Create new version
                new_version = version or self._get_next_version(existing.version)
                metadata = existing.copy()
                metadata.create_new_version(
                    version=new_version,
                    description=description or f"Update to {metadata.name}",
                    is_production=is_production
                )
                
                # Update metadata
                for k, v in kwargs.items():
                    if hasattr(metadata, k):
                        setattr(metadata, k, v)
                
                # Log the event
                self._log_event(
                    event_type=FeatureStoreEventType.FEATURE_VERSION_CREATED,
                    feature_name=metadata.name,
                    version=new_version,
                    metadata={
                        'description': description,
                        'is_production': is_production,
                        'tags': tags or []
                    }
                )
            else:
                # New feature
                if version:
                    metadata.version = version
                
                # Set initial version
                metadata.create_new_version(
                    version=version or "1.0.0",
                    description=description or f"Initial version of {metadata.name}",
                    is_production=is_production
                )
                
                # Set additional metadata
                metadata.created_by = kwargs.get('created_by', 'system')
                metadata.updated_by = kwargs.get('updated_by', metadata.created_by)
                
                # Log the event
                self._log_event(
                    event_type=FeatureStoreEventType.FEATURE_CREATED,
                    feature_name=metadata.name,
                    version=metadata.version,
                    metadata={
                        'description': description,
                        'is_production': is_production,
                        'tags': tags or []
                    }
                )
            
            # Save metadata
            return self._save_metadata(metadata)
            
        except Exception as e:
            logger.error(f"Error registering feature {metadata.name if hasattr(metadata, 'name') else 'unknown'}: {e}", exc_info=True)
            return False
    
    def _get_next_version(self, current_version: str) -> str:
        """Generate the next version number based on the current version."""
        try:
            major, minor, patch = map(int, current_version.split('.'))
            return f"{major}.{minor}.{patch + 1}"
        except (ValueError, AttributeError):
            return "1.0.0"
    
    def get_feature_metadata(self, name: str, use_cache: bool = True, version: Optional[str] = None) -> Optional[FeatureMetadata]:
        """Get feature metadata, optionally for a specific version.
        
        Args:
            name: Name of the feature
            use_cache: Whether to use cached metadata
            version: Optional specific version to get metadata for
            
        Returns:
            FeatureMetadata or None if not found
        """
        try:
            # Check cache first if enabled
            if use_cache and name in self._metadata_cache:
                metadata = self._metadata_cache[name]
            else:
                # Load from storage
                meta_dict = self.storage.load_metadata(name)
                if not meta_dict:
                    return None
                metadata = FeatureMetadata.from_dict(meta_dict)
                with self._cache_lock:
                    self._metadata_cache[name] = metadata
            
            # Return specific version if requested
            if version is not None:
                return metadata.get_version(version)
            return metadata
            
        except Exception as e:
            logger.error(f"Error getting metadata for {name}: {e}")
            return None
    
    def get_feature_data(
        self,
        name: str,
        version: Optional[str] = None,
        use_cache: bool = True,
        **kwargs
    ) -> Any:
        """Get feature data with caching and versioning support.
        
        Args:
            name: Name of the feature
            version: Optional version string (defaults to latest production version)
            use_cache: Whether to use cached data if available
            **kwargs: Additional arguments passed to the storage backend
            
        Returns:
            Feature data (DataFrame, array, or other serializable type)
        """
        cache_key = f"{name}:{version or 'latest'}"
        
        # Check cache first
        if use_cache and cache_key in self._data_cache:
            return self._data_cache[cache_key]
        
        try:
            # Get metadata to resolve version if not specified
            metadata = self.get_feature_metadata(name, use_cache=use_cache)
            if not metadata:
                raise ValueError(f"Feature {name} not found")
            
            # Resolve version if not specified
            if version is None:
                version = metadata.get_latest_production_version()
                if not version:
                    version = metadata.get_latest_version()
                    if not version:
                        raise ValueError(f"No versions found for feature {name}")
            
            # Load data from storage
            data = self.storage.load_data(name, version, **kwargs)
            if data is None:
                raise ValueError(f"No data found for {name} v{version}")
            
            # Update statistics
            self._update_feature_stats(metadata, version, data)
            
            # Update cache
            if use_cache:
                with self._cache_lock:
                    self._data_cache[cache_key] = data
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading data for {name} v{version or 'latest'}: {e}")
            raise
    
    def save_feature_data(
        self,
        name: str,
        data: Any,
        version: Optional[str] = None,
        description: str = "",
        is_production: bool = False,
        **kwargs
    ) -> bool:
        """Save feature data with versioning support.
        
        Args:
            name: Name of the feature
            data: Feature data (DataFrame, array, or other serializable type)
            version: Optional version string (auto-increments if None)
            description: Description of this version
            is_production: Whether to mark this version as production
            **kwargs: Additional arguments passed to the storage backend
            
        Returns:
            bool: True if save was successful
        """
        try:
            # Get or create metadata
            metadata = self.get_feature_metadata(name, use_cache=False)
            
            if metadata is None:
                # Create new feature
                metadata = FeatureMetadata(
                    name=name,
                    feature_type=self._infer_feature_type(data),
                    created_by=kwargs.get('created_by', 'system'),
                    updated_by=kwargs.get('updated_by', 'system')
                )
                version = version or "1.0.0"
                metadata.create_new_version(
                    version=version,
                    description=description or f"Initial version of {name}",
                    is_production=is_production
                )
            else:
                # Update existing feature
                version = version or self._get_next_version(metadata.version)
                metadata.create_new_version(
                    version=version,
                    description=description or f"Update to {name}",
                    is_production=is_production
                )
            
            # Validate data against schema if defined
            if metadata.schema:
                self._validate_data_against_schema(data, metadata.schema)
            
            # Save data
            success = self.storage.save_data(
                feature_name=name,
                version=version,
                data=data,
                **kwargs
            )
            
            if not success:
                raise RuntimeError(f"Failed to save data for {name} v{version}")
            
            # Update statistics
            self._update_feature_stats(metadata, version, data)
            
            # Save metadata
            metadata.updated_at = datetime.utcnow()
            metadata.updated_by = kwargs.get('updated_by', 'system')
            
            if not self._save_metadata(metadata):
                raise RuntimeError(f"Failed to save metadata for {name}")
            
            # Log the event
            self._log_event(
                event_type=FeatureStoreEventType.FEATURE_DATA_SAVED,
                feature_name=name,
                version=version,
                metadata={
                    'description': description,
                    'is_production': is_production,
                    'data_type': str(type(data)),
                    'data_shape': getattr(data, 'shape', None),
                    'data_columns': list(data.columns) if hasattr(data, 'columns') else None
                }
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving data for {name} v{version or 'new'}: {e}", exc_info=True)
            # Clean up partial saves
            if 'version' in locals():
                self.storage.delete_data(name, version)
            return False
    
    def _infer_feature_type(self, data: Any) -> FeatureType:
        """Infer feature type from data."""
        if isinstance(data, pd.DataFrame):
            return FeatureType.TABULAR
        elif isinstance(data, np.ndarray):
            if len(data.shape) == 1:
                return FeatureType.NUMERICAL
            elif len(data.shape) == 2:
                return FeatureType.EMBEDDING
            else:
                return FeatureType.TENSOR
        elif isinstance(data, (list, dict)):
            return FeatureType.CATEGORICAL
        else:
            return FeatureType.UNKNOWN
    
    def _validate_data_against_schema(self, data: Any, schema: Dict[str, Any]) -> None:
        """Validate data against schema."""
        # Basic schema validation
        if not schema:
            return
            
        if 'required_columns' in schema and hasattr(data, 'columns'):
            missing = set(schema['required_columns']) - set(data.columns)
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
        
        # Add more schema validation as needed
    
    def _update_feature_stats(self, metadata: FeatureMetadata, version: str, data: Any) -> None:
        """Update feature statistics."""
        try:
            if not isinstance(data, (pd.DataFrame, np.ndarray)):
                return
                
            stats = FeatureStatistics()
            
            if isinstance(data, pd.DataFrame):
                stats.row_count = len(data)
                stats.column_count = len(data.columns)
                stats.feature_type = FeatureType.TABULAR
                
                # Basic statistics for numeric columns
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    stats.numeric_stats = {
                        col: {
                            'min': float(data[col].min()),
                            'max': float(data[col].max()),
                            'mean': float(data[col].mean()),
                            'std': float(data[col].std()),
                            'null_count': int(data[col].isnull().sum())
                        }
                        for col in numeric_cols
                    }
                
                # Basic statistics for categorical columns
                cat_cols = data.select_dtypes(include=['object', 'category']).columns
                if len(cat_cols) > 0:
                    stats.categorical_stats = {
                        col: {
                            'unique_count': int(data[col].nunique()),
                            'top_value': data[col].mode().iloc[0] if not data[col].empty else None,
                            'null_count': int(data[col].isnull().sum())
                        }
                        for col in cat_cols
                    }
            
            elif isinstance(data, np.ndarray):
                stats.feature_type = FeatureType.EMBEDDING if len(data.shape) > 1 else FeatureType.NUMERICAL
                stats.row_count = data.shape[0]
                if len(data.shape) > 1:
                    stats.column_count = data.shape[1]
                
                if np.issubdtype(data.dtype, np.number):
                    stats.numeric_stats = {
                        'all': {
                            'min': float(np.nanmin(data)),
                            'max': float(np.nanmax(data)),
                            'mean': float(np.nanmean(data)),
                            'std': float(np.nanstd(data)),
                            'null_count': int(np.isnan(data).sum())
                        }
                    }
            
            # Update metadata
            metadata.update_statistics(version, stats)
            self._save_metadata(metadata)
            
        except Exception as e:
            logger.warning(f"Error updating statistics for {metadata.name} v{version}: {e}")
    
    def delete_feature(self, name: str, version: Optional[str] = None) -> bool:
        """Delete a feature or a specific version.
        
        Args:
            name: Name of the feature
            version: Optional version to delete (deletes all versions if None)
            
        Returns:
            bool: True if deletion was successful
        """
        try:
            if version:
                # Delete specific version
                success = self.storage.delete_data(name, version)
                if success:
                    # Update metadata
                    metadata = self.get_feature_metadata(name, use_cache=False)
                    if metadata:
                        if version in metadata.versions:
                            del metadata.versions[version]
                            # If no versions left, delete the feature entirely
                            if not metadata.versions:
                                return self.delete_feature(name)
                            return self._save_metadata(metadata)
                return success
            else:
                # Delete all versions and metadata
                success = True
                versions = self.storage.list_versions(name)
                for v in versions:
                    if not self.storage.delete_data(name, v):
                        success = False
                
                # Delete metadata
                if success and self.storage.delete_metadata(name):
                    with self._cache_lock:
                        if name in self._metadata_cache:
                            del self._metadata_cache[name]
                    self._log_event(
                        event_type=FeatureStoreEventType.FEATURE_DELETED,
                        feature_name=name,
                        version='all',
                        metadata={'action': 'feature_deleted'}
                    )
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Error deleting feature {name} v{version or 'all'}: {e}")
            return False
            
    def list_features(self, pattern: str = "*") -> List[str]:
        """List all features matching a pattern."""
        try:
            features = self.storage.list_features()
            if pattern != "*":
                import fnmatch
                features = [f for f in features if fnmatch.fnmatch(f, pattern)]
            return features
        except Exception as e:
            logger.error(f"Error listing features: {e}")
            return []
    
    def list_versions(self, name: str) -> List[Dict[str, Any]]:
        """List all versions of a feature with metadata."""
        try:
            metadata = self.get_feature_metadata(name)
            if not metadata:
                return []
                
            versions = []
            for version, version_meta in metadata.versions.items():
                versions.append({
                    'version': version,
                    'created_at': version_meta.created_at.isoformat() if version_meta.created_at else None,
                    'is_production': version_meta.is_production,
                    'description': version_meta.description,
                    'tags': version_meta.tags,
                    'statistics': metadata.statistics.get(version, {})
                })
            
            # Sort by version (newest first)
            return sorted(versions, key=lambda x: x['created_at'] or '', reverse=True)
            
        except Exception as e:
            logger.error(f"Error listing versions for {name}: {e}")
            return []
    
    def promote_to_production(self, name: str, version: str) -> bool:
        """Promote a feature version to production."""
        try:
            metadata = self.get_feature_metadata(name, use_cache=False)
            if not metadata:
                raise ValueError(f"Feature {name} not found")
            
            # Demote current production version if any
            for v in metadata.versions.values():
                if v.is_production:
                    v.is_production = False
            
            # Promote the specified version
            if version not in metadata.versions:
                raise ValueError(f"Version {version} not found for feature {name}")
                
            metadata.versions[version].is_production = True
            metadata.updated_at = datetime.utcnow()
            
            # Save changes
            if self._save_metadata(metadata):
                self._log_event(
                    event_type=FeatureStoreEventType.FEATURE_PROMOTED,
                    feature_name=name,
                    version=version,
                    metadata={'action': 'promoted_to_production'}
                )
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error promoting {name} v{version} to production: {e}")
            return False
    
    def get_production_version(self, name: str) -> Optional[str]:
        """Get the current production version of a feature."""
        metadata = self.get_feature_metadata(name)
        if not metadata:
            return None
            
        for version, version_meta in metadata.versions.items():
            if version_meta.is_production:
                return version
        return None
    
    def get_latest_version(self, name: str) -> Optional[str]:
        """Get the latest version of a feature."""
        versions = self.list_versions(name)
        if not versions:
            return None
        return versions[0]['version']
    
    def get_feature_statistics(self, name: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get statistics for a feature version."""
        metadata = self.get_feature_metadata(name)
        if not metadata:
            return None
            
        if version is None:
            version = self.get_production_version(name) or self.get_latest_version(name)
            if not version:
                return None
                
        if version not in metadata.statistics:
            # Try to load data to compute statistics
            try:
                data = self.get_feature_data(name, version, use_cache=False)
                if data is not None:
                    self._update_feature_stats(metadata, version, data)
            except Exception as e:
                logger.warning(f"Error computing statistics for {name} v{version}: {e}")
        
        stats = metadata.statistics.get(version)
        return stats.to_dict() if stats else None
    
    def search_features(
        self,
        query: Optional[str] = None,
        tags: Optional[List[str]] = None,
        feature_type: Optional[FeatureType] = None,
        min_version: Optional[str] = None,
        max_version: Optional[str] = None,
        created_after: Optional[datetime] = None,
        created_before: Optional[datetime] = None,
        updated_after: Optional[datetime] = None,
        updated_before: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Search for features based on various criteria."""
        try:
            # Get all features
            all_features = self.storage.list_features()
            
            # Apply filters
            results = []
            for name in all_features[offset:offset + limit]:
                try:
                    metadata = self.get_feature_metadata(name)
                    if not metadata:
                        continue
                    
                    # Apply filters
                    if feature_type and metadata.feature_type != feature_type:
                        continue
                        
                    if tags and not any(tag in (metadata.tags or []) for tag in tags):
                        continue
                        
                    if created_after and metadata.created_at < created_after:
                        continue
                        
                    if created_before and metadata.created_at > created_before:
                        continue
                        
                    if updated_after and metadata.updated_at < updated_after:
                        continue
                        
                    if updated_before and metadata.updated_at > updated_before:
                        continue
                    
                    # Apply version filters
                    versions = []
                    for version, version_meta in metadata.versions.items():
                        if min_version and version < min_version:
                            continue
                        if max_version and version > max_version:
                            continue
                            
                        versions.append({
                            'version': version,
                            'is_production': version_meta.is_production,
                            'created_at': version_meta.created_at.isoformat() if version_meta.created_at else None,
                            'description': version_meta.description,
                            'tags': version_meta.tags
                        })
                    
                    if not versions:
                        continue
                    
                    # Apply text search
                    if query and query.lower() not in name.lower() and \
                       query.lower() not in (metadata.description or '').lower() and \
                       not any(query.lower() in tag.lower() for tag in metadata.tags or []):
                        continue
                    
                    # Add to results
                    results.append({
                        'name': name,
                        'feature_type': metadata.feature_type.value,
                        'description': metadata.description,
                        'tags': metadata.tags or [],
                        'created_at': metadata.created_at.isoformat() if metadata.created_at else None,
                        'updated_at': metadata.updated_at.isoformat() if metadata.updated_at else None,
                        'created_by': metadata.created_by,
                        'updated_by': metadata.updated_by,
                        'versions': sorted(versions, key=lambda x: x['version'], reverse=True),
                        'production_version': next((v['version'] for v in versions if v['is_production']), None)
                    })
                    
                except Exception as e:
                    logger.warning(f"Error processing feature {name}: {e}")
            
            # Sort results
            results.sort(key=lambda x: x['name'])
            
            return {
                'total': len(results),
                'offset': offset,
                'limit': limit,
                'features': results
            }
            
        except Exception as e:
            logger.error(f"Error searching features: {e}")
            return {
                'total': 0,
                'offset': offset,
                'limit': limit,
                'features': []
            }
    
    def write_feature_values(
        self,
        feature_name: str,
        entity_id: str,
        values: Dict[datetime, Any],
        metadata: Optional[Dict] = None
    ) -> bool:
        """Write feature values to the store."""
        try:
            feature_meta = self.get_feature_metadata(feature_name)
            if not feature_meta:
                raise ValueError(f"Feature {feature_name} not found")
                
            # Create directory structure
            feature_dir = self.data_path / feature_name
            feature_dir.mkdir(exist_ok=True)
            
            # Convert values to DataFrame
            df = pd.DataFrame([
                {
                    'entity_id': entity_id,
                    'timestamp': ts,
                    'value': self._serialize_value(val),
                    'metadata': metadata or {}
                }
                for ts, val in values.items()
            ])
            
            if df.empty:
                return True
                
            # Write to Parquet
            table = pa.Table.from_pandas(df)
            
            # Use timestamp for partitioning
            timestamp = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"{entity_id}_{timestamp}_{str(uuid.uuid4())[:8]}.parquet"
            output_file = feature_dir / filename
            
            pq.write_table(table, output_file)
            return True
            
        except Exception as e:
            logger.error(f"Error writing feature values for {feature_name}: {str(e)}")
            return False
    
    def get_feature_values(
        self,
        feature_name: str,
        entity_ids: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 10000
    ) -> pd.DataFrame:
        """Get feature values from the store."""
        try:
            feature_dir = self.data_path / feature_name
            if not feature_dir.exists():
                return pd.DataFrame()
                
            # Read all Parquet files
            dataset = ds.dataset(
                str(feature_dir),
                format="parquet",
                partitioning="hive"
            )
            
            # Build filters
            filters = []
            if entity_ids:
                filters.append(ds.field("entity_id").isin(entity_ids))
            if start_time:
                filters.append(ds.field("timestamp") >= start_time)
            if end_time:
                filters.append(ds.field("timestamp") <= end_time)
                
            # Apply filters if any
            if filters:
                dataset = dataset.filter(ds.all(*filters))
                
            # Convert to table
            table = dataset.head(limit).to_table()
            
            # Convert to DataFrame
            df = table.to_pandas()
            
            # Deserialize values
            if not df.empty and 'value' in df.columns:
                df['value'] = df['value'].apply(self._deserialize_value)
                
            return df
            
        except Exception as e:
            logger.error(f"Error reading feature values for {feature_name}: {str(e)}")
            return pd.DataFrame()
    
    def create_feature_set(
        self,
        name: str,
        features: List[str],
        description: str = "",
        tags: Optional[List[str]] = None
    ) -> bool:
        """Create a new feature set."""
        try:
            # Validate features
            for feature in features:
                if feature not in self._metadata_cache:
                    raise ValueError(f"Feature {feature} not found")
            
            # Create feature set metadata
            feature_set = {
                'name': name,
                'features': features,
                'description': description,
                'tags': tags or [],
                'created_at': datetime.utcnow().isoformat(),
                'updated_at': datetime.utcnow().isoformat()
            }
            
            # Save to disk
            feature_set_file = self.metadata_path / f"featureset_{name}.json"
            with open(feature_set_file, 'w') as f:
                json.dump(feature_set, f, indent=2)
                
            return True
            
        except Exception as e:
            logger.error(f"Error creating feature set {name}: {str(e)}")
            return False
    
    def get_feature_set(self, name: str) -> Optional[Dict]:
        """Get a feature set by name."""
        try:
            feature_set_file = self.metadata_path / f"featureset_{name}.json"
            if not feature_set_file.exists():
                return None
                
            with open(feature_set_file, 'r') as f:
                return json.load(f)
                
        except Exception as e:
            logger.error(f"Error reading feature set {name}: {str(e)}")
            return None
    
    def get_feature_set_data(
        self,
        feature_set_name: str,
        entity_ids: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 10000
    ) -> pd.DataFrame:
        """Get data for all features in a feature set."""
        feature_set = self.get_feature_set(feature_set_name)
        if not feature_set:
            raise ValueError(f"Feature set {feature_set_name} not found")
            
        # Collect data for each feature
        dfs = []
        for feature in feature_set['features']:
            df = self.get_feature_values(
                feature_name=feature,
                entity_ids=entity_ids,
                start_time=start_time,
                end_time=end_time,
                limit=limit
            )
            
            if not df.empty:
                # Pivot to wide format
                df = df.pivot(
                    index=['entity_id', 'timestamp'],
                    columns='feature_name',
                    values='value'
                ).reset_index()
                dfs.append(df)
        
        # Merge all features
        if not dfs:
            return pd.DataFrame()
            
        result = pd.concat(dfs, axis=0, ignore_index=True)
        return result
    
    def _serialize_value(self, value: Any) -> Any:
        """Serialize a value for storage."""
        if isinstance(value, (np.integer, np.floating, np.bool_)):
            return float(value) if isinstance(value, np.floating) else int(value)
        elif isinstance(value, (np.ndarray, list, tuple)):
            return [self._serialize_value(v) for v in value]
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        elif hasattr(value, 'tolist'):
            return value.tolist()
        return value
    
    def _deserialize_value(self, value: Any) -> Any:
        """Deserialize a stored value."""
        if isinstance(value, (list, dict)) or pd.isna(value):
            return value
        return float(value) if '.' in str(value) else int(value)

    def _init_online_serving(self) -> None:
        """Initialize the FastAPI app for online serving."""
        from fastapi import FastAPI, HTTPException, Depends, status, Query
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import JSONResponse, FileResponse
        from fastapi.staticfiles import StaticFiles
        from fastapi.openapi.docs import get_swagger_ui_html
        from fastapi.openapi.utils import get_openapi
        from pydantic import BaseModel
        from typing import List as TList, Optional as TOptional, Dict as TDict, Any as TAny, Union as TUnion
        import uvicorn
        import threading
        import tempfile
        import os
        
        # Create FastAPI app
        app = FastAPI(
            title="Feature Store API",
            description="REST API for feature store operations",
            version="1.0.0",
            docs_url=None,  # We'll serve docs at /docs
            redoc_url=None  # We'll serve redoc at /redoc
        )
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Serve static files for the web UI
        static_dir = os.path.join(tempfile.gettempdir(), "feature_store_ui")
        os.makedirs(static_dir, exist_ok=True)
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
        
        # Custom OpenAPI schema
        def custom_openapi():
            if app.openapi_schema:
                return app.openapi_schema
                
            openapi_schema = get_openapi(
                title="Feature Store API",
                version="1.0.0",
                description="REST API for feature store operations",
                routes=app.routes,
            )
            
            # Add security definitions
            openapi_schema["components"]["securitySchemes"] = {
                "BearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "bearerFormat": "JWT"
                }
            }
            
            # Add security to all endpoints
            for path in openapi_schema.get("paths", {}).values():
                for method in path.values():
                    method["security"] = [{"BearerAuth": []}]
            
            app.openapi_schema = openapi_schema
            return app.openapi_schema
        
        app.openapi = custom_openapi
        
        # Custom docs endpoints to include auth
        @app.get("/docs", include_in_schema=False)
        async def custom_swagger_ui_html():
            return get_swagger_ui_html(
                openapi_url=app.openapi_url,
                title=app.title + " - Swagger UI",
                oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
                swagger_js_url="/static/swagger-ui-bundle.js",
                swagger_css_url="/static/swagger-ui.css",
                swagger_favicon_url=None,
            )
        
        # API Models
        class FeatureRequest(BaseModel):
            name: str
            version: TOptional[str] = None
            params: TOptional[TDict[str, TAny]] = None
        
        class BatchFeatureRequest(BaseModel):
            features: TList[FeatureRequest]
        
        class FeatureResponse(BaseModel):
            name: str
            version: str
            data: TAny
            metadata: TOptional[TDict[str, TAny]] = None
        
        class ErrorResponse(BaseModel):
            error: str
            detail: TOptional[TDict[str, TAny]] = None
        
        # Helper functions
        def get_feature_store():
            return self
        
        # API Endpoints
        @app.get("/health", response_model=TDict[str, str], tags=["System"])
        async def health_check():
            """Health check endpoint"""
            return {"status": "healthy"}
        
        @app.get("/api/features", response_model=TDict[str, TAny], tags=["Features"])
        async def list_features(
            pattern: str = "*",
            limit: int = Query(100, ge=1, le=1000),
            offset: int = Query(0, ge=0),
            store: FeatureStore = Depends(get_feature_store)
        ):
            """List all features matching a pattern"""
            try:
                return store.search_features(
                    query=pattern if pattern != "*" else None,
                    limit=limit,
                    offset=offset
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/api/features/{feature_name}", response_model=FeatureResponse, responses={404: {"model": ErrorResponse}})
        async def get_feature(
            feature_name: str,
            version: TOptional[str] = None,
            store: FeatureStore = Depends(get_feature_store)
        ):
            """Get feature data"""
            try:
                # Get metadata first
                metadata = store.get_feature_metadata(feature_name, use_cache=True)
                if not metadata:
                    raise HTTPException(status_code=404, detail=f"Feature {feature_name} not found")
                
                # Resolve version if not specified
                if not version:
                    version = metadata.get_latest_production_version() or store.get_latest_version(feature_name)
                    if not version:
                        raise HTTPException(status_code=404, detail=f"No versions found for feature {feature_name}")
                
                # Get data
                data = store.get_feature_data(feature_name, version=version)
                if data is None:
                    raise HTTPException(status_code=404, detail=f"No data found for {feature_name} v{version}")
                
                # Convert data to JSON-serializable format
                if isinstance(data, (pd.DataFrame, pd.Series)):
                    data = data.to_dict(orient='records' if isinstance(data, pd.DataFrame) else 'list')
                elif isinstance(data, np.ndarray):
                    data = data.tolist()
                
                return FeatureResponse(
                    name=feature_name,
                    version=version,
                    data=data,
                    metadata={
                        'feature_type': metadata.feature_type.value,
                        'description': metadata.description,
                        'tags': metadata.tags,
                        'created_at': metadata.created_at.isoformat() if metadata.created_at else None,
                        'updated_at': metadata.updated_at.isoformat() if metadata.updated_at else None,
                        'statistics': store.get_feature_statistics(feature_name, version)
                    }
                )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error getting feature {feature_name}: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/api/features/batch", response_model=TList[FeatureResponse], tags=["Features"])
        async def get_batch_features(
            request: BatchFeatureRequest,
            store: FeatureStore = Depends(get_feature_store)
        ):
            """Get multiple features in a single request"""
            try:
                results = []
                for feature_req in request.features:
                    try:
                        # Get metadata first
                        metadata = store.get_feature_metadata(feature_req.name, use_cache=True)
                        if not metadata:
                            results.append({
                                'name': feature_req.name,
                                'error': f"Feature {feature_req.name} not found"
                            })
                            continue
                        
                        # Resolve version
                        version = feature_req.version
                        if not version:
                            version = metadata.get_latest_production_version() or store.get_latest_version(feature_req.name)
                            if not version:
                                results.append({
                                    'name': feature_req.name,
                                    'error': f"No versions found for feature {feature_req.name}"
                                })
                                continue
                        
                        # Get data
                        data = store.get_feature_data(
                            feature_req.name, 
                            version=version,
                            **(feature_req.params or {})
                        )
                        
                        if data is None:
                            results.append({
                                'name': feature_req.name,
                                'version': version,
                                'error': f"No data found for version {version}"
                            })
                            continue
                        
                        # Convert data to JSON-serializable format
                        if isinstance(data, (pd.DataFrame, pd.Series)):
                            data = data.to_dict(orient='records' if isinstance(data, pd.DataFrame) else 'list')
                        elif isinstance(data, np.ndarray):
                            data = data.tolist()
                        
                        results.append({
                            'name': feature_req.name,
                            'version': version,
                            'data': data,
                            'metadata': {
                                'feature_type': metadata.feature_type.value,
                                'description': metadata.description,
                                'tags': metadata.tags,
                                'statistics': store.get_feature_statistics(feature_req.name, version)
                            }
                        })
                        
                    except Exception as e:
                        logger.error(f"Error processing feature {feature_req.name}: {e}", exc_info=True)
                        results.append({
                            'name': feature_req.name,
                            'version': feature_req.version or 'latest',
                            'error': str(e)
                        })
                
                return results
                
            except Exception as e:
                logger.error(f"Error in batch feature request: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/api/features/{feature_name}/promote/{version}", status_code=status.HTTP_204_NO_CONTENT, tags=["Features"])
        async def promote_feature_version(
            feature_name: str,
            version: str,
            store: FeatureStore = Depends(get_feature_store)
        ):
            """Promote a feature version to production"""
            try:
                if not store.promote_to_production(feature_name, version):
                    raise HTTPException(status_code=400, detail=f"Failed to promote {feature_name} v{version}")
                return None
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))
            except Exception as e:
                logger.error(f"Error promoting {feature_name} v{version}: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
        
        # Error handlers
        @app.exception_handler(HTTPException)
        async def http_exception_handler(request, exc):
            return JSONResponse(
                status_code=exc.status_code,
                content={"error": exc.detail}
            )
            
        @app.exception_handler(Exception)
        async def global_exception_handler(request, exc):
            logger.error(f"Unhandled exception: {exc}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error"}
            )
        
        # Store the app instance
        self._fastapi_app = app
        
        # Start the server in a background thread
        def run_server():
            uvicorn.run(
                app,
                host="0.0.0.0",
                port=self.online_serving_port,
                log_level="info",
                access_log=True
            )
        
        self._server_thread = threading.Thread(
            target=run_server,
            daemon=True,
            name="feature_store_server"
        )
        self._server_thread.start()
        
        logger.info(f"Feature store API server started on port {self.online_serving_port}")
    
    def stop_online_serving(self) -> None:
        """Stop the online serving API server."""
        if hasattr(self, '_server_thread') and self._server_thread.is_alive():
            # This is a simple implementation - in production you'd want to properly shut down uvicorn
            logger.info("Stopping feature store API server...")
            # Note: This is a simple implementation. In production, you'd want to properly
            # shut down the uvicorn server using its shutdown hooks.
            self._fastapi_app = None
            logger.info("Feature store API server stopped")
            
    def export_feature(
        self,
        name: str,
        version: Optional[str] = None,
        output_format: str = 'parquet',
        output_path: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> Optional[Union[str, bytes]]:
        """Export feature data to a file or return as bytes.
        
        Args:
            name: Name of the feature
            version: Version to export (defaults to latest production version)
            output_format: Output format ('parquet', 'csv', 'json', 'pickle', 'hdf5')
            output_path: Optional path to save the exported file
            **kwargs: Additional arguments for the export function
            
        Returns:
            Path to the exported file or bytes if output_path is None
        """
        try:
            # Get the data
            data = self.get_feature_data(name, version=version, use_cache=False)
            if data is None:
                raise ValueError(f"No data found for feature {name}")
            
            # Determine output path
            if output_path is None:
                output_path = Path(tempfile.mktemp(suffix=f".{output_format}"))
                delete_after = True
            else:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                delete_after = False
            
            # Export the data
            result_path = FeatureExporter.export_feature(
                data=data,
                output_path=output_path,
                output_format=output_format,
                **kwargs
            )
            
            # Read and return bytes if no output path was provided
            if delete_after:
                with open(result_path, 'rb') as f:
                    result = f.read()
                Path(result_path).unlink()
                return result
                
            return result_path
            
        except Exception as e:
            logger.error(f"Error exporting feature {name}: {e}", exc_info=True)
            raise
            
    def import_feature(
        self,
        name: str,
        data: Union[str, Path, bytes, Dict[str, Any], pd.DataFrame, np.ndarray],
        input_format: Optional[str] = None,
        version: Optional[str] = None,
        description: str = "",
        is_production: bool = False,
        **kwargs
    ) -> bool:
        """Import feature data from a file or object.
        
        Args:
            name: Name of the feature
            data: Input data (file path, bytes, or data object)
            input_format: Input format ('parquet', 'csv', 'json', 'pickle', 'hdf5')
                         If None, will try to infer from file extension or data type
            version: Version to create (defaults to auto-increment)
            description: Description of this version
            is_production: Whether to mark this version as production
            **kwargs: Additional arguments for the import function
            
        Returns:
            bool: True if import was successful
        """
        try:
            # Import the data
            imported_data = FeatureImporter.import_data(
                data=data,
                input_format=input_format,
                **kwargs
            )
            
            # Save the feature
            return self.save_feature_data(
                name=name,
                data=imported_data,
                version=version,
                description=description,
                is_production=is_production
            )
            
        except Exception as e:
            logger.error(f"Error importing feature {name}: {e}", exc_info=True)
            return False
            
    def backup(
        self, 
        output_path: Union[str, Path], 
        include_data: bool = True,
        compression: str = 'zip'
    ) -> Path:
        """Create a backup of the feature store.
        
        Args:
            output_path: Path to save the backup file or directory
            include_data: Whether to include feature data in the backup
            compression: Compression format ('zip', 'tar', 'gztar', 'bztar', 'xztar', or None)
            
        Returns:
            Path to the created backup file or directory
        """
        return FeatureBackup.create_backup(
            feature_store=self,
            output_path=output_path,
            include_data=include_data,
            compression=compression
        )
        
    @classmethod
    def restore(
        cls,
        backup_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        base_path: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> 'FeatureStore':
        """Restore a feature store from a backup.
        
        Args:
            backup_path: Path to the backup file or directory
            output_path: Path to restore to (defaults to original path or base_path)
            base_path: Base path for the new feature store (if output_path not provided)
            **kwargs: Additional arguments for FeatureStore initialization
            
        Returns:
            FeatureStore: The restored feature store
        """
        # Determine output path
        if output_path is None:
            if base_path is None:
                output_path = Path(backup_path).parent / f"restored_{Path(backup_path).name}"
            else:
                output_path = Path(base_path)
        
        # Initialize feature store
        store = cls(base_path=output_path, **kwargs)
        
        # Restore from backup
        success = FeatureBackup.restore_backup(backup_path, store, include_data=True)
        if not success:
            raise RuntimeError("Failed to restore feature store from backup")
            
        return store

class OnlineFeatureStore(FeatureStore):
    """In-memory feature store for online serving."""
    
    def __init__(self, base_path: str):
        super().__init__(base_path)
        self._online_cache: Dict[str, Dict[str, Any]] = {}
    
    def load_features(self, feature_names: List[str]):
        """Load features into the online store."""
        for feature in feature_names:
            # Load recent data for online serving
            df = self.get_feature_values(
                feature_name=feature,
                limit=100000  # Adjust based on memory constraints
            )
            
            if not df.empty:
                # Index by entity_id and timestamp
                self._online_cache[feature] = df.set_index(['entity_id', 'timestamp'])['value'].to_dict()
    
    def get_online_features(
        self,
        entity_ids: List[str],
        feature_names: List[str],
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get feature values for online serving."""
        if timestamp is None:
            timestamp = datetime.utcnow()
            
        results = {}
        for entity_id in entity_ids:
            entity_results = {}
            for feature in feature_names:
                if feature in self._online_cache:
                    # Find the most recent value before the given timestamp
                    latest = None
                    for ts, val in sorted(
                        ((k[1], v) for k, v in self._online_cache[feature].items() 
                         if k[0] == entity_id and k[1] <= timestamp),
                        key=lambda x: x[0],
                        reverse=True
                    ):
                        latest = val
                        break
                    entity_results[feature] = latest
            results[entity_id] = entity_results
            
        return results
