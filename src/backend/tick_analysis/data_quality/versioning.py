"""
Data versioning and lineage tracking system.
"""

import hashlib
import json
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import uuid
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class VersioningBackend(Enum):
    FILESYSTEM = "filesystem"
    DATABASE = "database"
    MEMORY = "memory"

class DataOperation(Enum):
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    TRANSFORM = "transform"
    AGGREGATE = "aggregate"
    FILTER = "filter"

@dataclass
class DataVersion:
    """Represents a version of a dataset."""
    version_id: str
    dataset_id: str
    timestamp: datetime
    operation: DataOperation
    checksum: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_versions: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    created_by: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['operation'] = self.operation.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataVersion':
        """Create from dictionary."""
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['operation'] = DataOperation(data['operation'])
        return cls(**data)

class DataVersioning:
    """
    Data versioning system with support for:
    - Version tracking
    - Lineage/provenance
    - Metadata management
    - Checksum validation
    """
    
    def __init__(self, backend: VersioningBackend = VersioningBackend.FILESYSTEM, **backend_kwargs):
        """
        Initialize the versioning system.
        
        Args:
            backend: Backend storage type
            **backend_kwargs: Backend-specific configuration
        """
        self.backend = backend
        self.backend_kwargs = backend_kwargs
        self.versions: Dict[str, DataVersion] = {}
        self._init_backend()
    
    def _init_backend(self) -> None:
        """Initialize the selected backend."""
        if self.backend == VersioningBackend.MEMORY:
            self.versions = {}
        elif self.backend == VersioningBackend.FILESYSTEM:
            import os
            self.storage_path = self.backend_kwargs.get('storage_path', './data/versions')
            os.makedirs(self.storage_path, exist_ok=True)
        elif self.backend == VersioningBackend.DATABASE:
            # Initialize database connection
            self.db_conn = self.backend_kwargs.get('connection')
            self._init_database()
    
    def _init_database(self) -> None:
        """Initialize database tables if they don't exist."""
        if self.backend != VersioningBackend.DATABASE or not self.db_conn:
            return
            
        with self.db_conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS data_versions (
                    version_id TEXT PRIMARY KEY,
                    dataset_id TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    operation TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    metadata JSONB,
                    parent_versions TEXT[],
                    tags TEXT[],
                    created_by TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_data_versions_dataset_id ON data_versions(dataset_id);
                CREATE INDEX IF NOT EXISTS idx_data_versions_timestamp ON data_versions(timestamp);
            """)
            self.db_conn.commit()
    
    def create_version(
        self,
        dataset_id: str,
        data: Any,
        operation: DataOperation = DataOperation.CREATE,
        parent_versions: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        created_by: Optional[str] = None
    ) -> DataVersion:
        """
        Create a new version of a dataset.
        
        Args:
            dataset_id: Unique identifier for the dataset
            data: The data to version
            operation: Type of operation being performed
            parent_versions: List of parent version IDs
            metadata: Additional metadata
            tags: Version tags
            created_by: User or system creating the version
            
        Returns:
            DataVersion: The created version
        """
        checksum = self._calculate_checksum(data)
        version_id = str(uuid.uuid4())
        timestamp = datetime.utcnow()
        
        version = DataVersion(
            version_id=version_id,
            dataset_id=dataset_id,
            timestamp=timestamp,
            operation=operation,
            checksum=checksum,
            metadata=metadata or {},
            parent_versions=parent_versions or [],
            tags=tags or [],
            created_by=created_by
        )
        
        self._store_version(version)
        return version
    
    def get_version(self, version_id: str) -> Optional[DataVersion]:
        """Retrieve a version by ID."""
        if self.backend == VersioningBackend.MEMORY:
            return self.versions.get(version_id)
        elif self.backend == VersioningBackend.FILESYSTEM:
            return self._load_version_from_fs(version_id)
        elif self.backend == VersioningBackend.DATABASE:
            return self._load_version_from_db(version_id)
        return None
    
    def list_versions(
        self,
        dataset_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        tags: Optional[List[str]] = None
    ) -> List[DataVersion]:
        """List versions with optional filtering."""
        if self.backend == VersioningBackend.MEMORY:
            versions = list(self.versions.values())
        elif self.backend == VersioningBackend.FILESYSTEM:
            versions = self._list_versions_fs()
        elif self.backend == VersioningBackend.DATABASE:
            versions = self._list_versions_db(dataset_id, limit, offset, tags)
        else:
            versions = []
        
        # Apply filters
        if dataset_id:
            versions = [v for v in versions if v.dataset_id == dataset_id]
        if tags:
            versions = [v for v in versions if any(tag in v.tags for tag in tags)]
        
        return versions[offset:offset+limit]
    
    def get_lineage(self, version_id: str, max_depth: int = 5) -> Dict[str, Any]:
        """Get the lineage/provenance of a version."""
        version = self.get_version(version_id)
        if not version:
            return {}
        
        lineage = {
            'version': version.to_dict(),
            'parents': [],
            'children': []
        }
        
        # Get parents
        if version.parent_versions and max_depth > 0:
            for parent_id in version.parent_versions:
                parent_lineage = self.get_lineage(parent_id, max_depth - 1)
                if parent_lineage:
                    lineage['parents'].append(parent_lineage)
        
        # Get children (would require reverse index in real implementation)
        if self.backend == VersioningBackend.DATABASE:
            children = self._get_children_from_db(version_id)
            for child_id in children:
                child_lineage = self.get_lineage(child_id, max_depth - 1)
                if child_lineage:
                    lineage['children'].append(child_lineage)
        
        return lineage
    
    def _calculate_checksum(self, data: Any) -> str:
        """Calculate checksum for data."""
        if isinstance(data, (str, bytes)):
            if isinstance(data, str):
                data = data.encode('utf-8')
        elif hasattr(data, 'read'):
            # File-like object
            data = data.read()
        else:
            # Convert to JSON string
            data = json.dumps(data, sort_keys=True).encode('utf-8')
        
        return hashlib.sha256(data).hexdigest()
    
    def _store_version(self, version: DataVersion) -> None:
        """Store a version in the backend."""
        if self.backend == VersioningBackend.MEMORY:
            self.versions[version.version_id] = version
        elif self.backend == VersioningBackend.FILESYSTEM:
            self._store_version_fs(version)
        elif self.backend == VersioningBackend.DATABASE:
            self._store_version_db(version)
    
    # Filesystem backend methods
    def _store_version_fs(self, version: DataVersion) -> None:
        """Store version in filesystem."""
        import os
        import json
        
        version_dir = os.path.join(self.storage_path, version.dataset_id)
        os.makedirs(version_dir, exist_ok=True)
        
        version_file = os.path.join(version_dir, f"{version.version_id}.json")
        with open(version_file, 'w') as f:
            json.dump(version.to_dict(), f, indent=2)
    
    def _load_version_from_fs(self, version_id: str) -> Optional[DataVersion]:
        """Load version from filesystem."""
        import os
        import json
        import glob
        
        # Find the version file (inefficient for large numbers of files)
        pattern = os.path.join(self.storage_path, '**', f"{version_id}.json")
        matches = glob.glob(pattern, recursive=True)
        
        if not matches:
            return None
            
        with open(matches[0], 'r') as f:
            data = json.load(f)
            
        return DataVersion.from_dict(data)
    
    def _list_versions_fs(self) -> List[DataVersion]:
        """List all versions from filesystem."""
        import os
        import json
        import glob
        
        versions = []
        for version_file in glob.glob(os.path.join(self.storage_path, '**', '*.json'), recursive=True):
            try:
                with open(version_file, 'r') as f:
                    data = json.load(f)
                    versions.append(DataVersion.from_dict(data))
            except Exception as e:
                logger.warning(f"Error loading version from {version_file}: {e}")
        
        return sorted(versions, key=lambda v: v.timestamp, reverse=True)
    
    # Database backend methods
    def _store_version_db(self, version: DataVersion) -> None:
        """Store version in database."""
        if not self.db_conn:
            raise ValueError("Database connection not initialized")
            
        with self.db_conn.cursor() as cur:
            cur.execute("""
                INSERT INTO data_versions (
                    version_id, dataset_id, timestamp, operation, 
                    checksum, metadata, parent_versions, tags, created_by
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (version_id) DO UPDATE SET
                    dataset_id = EXCLUDED.dataset_id,
                    timestamp = EXCLUDED.timestamp,
                    operation = EXCLUDED.operation,
                    checksum = EXCLUDED.checksum,
                    metadata = EXCLUDED.metadata,
                    parent_versions = EXCLUDED.parent_versions,
                    tags = EXCLUDED.tags,
                    created_by = EXCLUDED.created_by;
            """, (
                version.version_id,
                version.dataset_id,
                version.timestamp,
                version.operation.value,
                version.checksum,
                json.dumps(version.metadata),
                version.parent_versions,
                version.tags,
                version.created_by
            ))
            self.db_conn.commit()
    
    def _load_version_from_db(self, version_id: str) -> Optional[DataVersion]:
        """Load version from database."""
        if not self.db_conn:
            return None
            
        with self.db_conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    version_id, dataset_id, timestamp, operation, 
                    checksum, metadata, parent_versions, tags, created_by
                FROM data_versions
                WHERE version_id = %s;
            """, (version_id,))
            
            row = cur.fetchone()
            if not row:
                return None
                
            return DataVersion(
                version_id=row[0],
                dataset_id=row[1],
                timestamp=row[2],
                operation=DataOperation(row[3]),
                checksum=row[4],
                metadata=row[5] or {},
                parent_versions=row[6] or [],
                tags=row[7] or [],
                created_by=row[8]
            )
    
    def _list_versions_db(
        self,
        dataset_id: Optional[str],
        limit: int,
        offset: int,
        tags: Optional[List[str]]
    ) -> List[DataVersion]:
        """List versions from database with filtering."""
        if not self.db_conn:
            return []
            
        query = """
            SELECT 
                version_id, dataset_id, timestamp, operation, 
                checksum, metadata, parent_versions, tags, created_by
            FROM data_versions
            WHERE 1=1
        """
        
        params = []
        
        if dataset_id:
            query += " AND dataset_id = %s"
            params.append(dataset_id)
            
        if tags:
            tag_conditions = []
            for tag in tags:
                query += f" AND %s = ANY(tags)"
                params.append(tag)
        
        query += " ORDER BY timestamp DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])
        
        with self.db_conn.cursor() as cur:
            cur.execute(query, tuple(params))
            rows = cur.fetchall()
            
        return [
            DataVersion(
                version_id=row[0],
                dataset_id=row[1],
                timestamp=row[2],
                operation=DataOperation(row[3]),
                checksum=row[4],
                metadata=row[5] or {},
                parent_versions=row[6] or [],
                tags=row[7] or [],
                created_by=row[8]
            )
            for row in rows
        ]
    
    def _get_children_from_db(self, version_id: str) -> List[str]:
        """Get children versions from database."""
        if not self.db_conn:
            return []
            
        with self.db_conn.cursor() as cur:
            cur.execute("""
                SELECT version_id 
                FROM data_versions 
                WHERE %s = ANY(parent_versions);
            """, (version_id,))
            return [row[0] for row in cur.fetchall()]
