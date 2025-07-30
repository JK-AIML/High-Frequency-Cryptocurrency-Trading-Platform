"""
Data Versioning and Lineage Module

This module provides comprehensive data versioning and lineage tracking capabilities,
including version management, metadata tracking, and lineage visualization.
"""

import logging
import asyncio
import json
import os
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import aiofiles
from enum import Enum

logger = logging.getLogger(__name__)

class VersionType(Enum):
    """Types of data versions."""
    SNAPSHOT = "snapshot"
    DELTA = "delta"
    MERGE = "merge"

@dataclass
class VersionMetadata:
    """Metadata for a data version."""
    version_id: str
    timestamp: datetime
    version_type: VersionType
    description: str
    author: str
    tags: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LineageNode:
    """Node in the data lineage graph."""
    version_id: str
    metadata: VersionMetadata
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    transformations: List[Dict[str, Any]] = field(default_factory=list)

class VersionStore:
    """Manages data versions and their storage."""
    
    def __init__(self, storage_path: str):
        """
        Initialize the version store.
        
        Args:
            storage_path: Path to store version data
        """
        self.storage_path = storage_path
        self.versions_path = os.path.join(storage_path, 'versions')
        self.metadata_path = os.path.join(storage_path, 'metadata')
        
        # Create storage directories
        os.makedirs(self.versions_path, exist_ok=True)
        os.makedirs(self.metadata_path, exist_ok=True)
    
    def _generate_version_id(self, data: Dict[str, Any]) -> str:
        """
        Generate a unique version ID for data.
        
        Args:
            data: Data to generate version ID for
            
        Returns:
            Unique version ID
        """
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    async def create_version(self,
                           data: Dict[str, Any],
                           metadata: VersionMetadata) -> str:
        """
        Create a new version of data.
        
        Args:
            data: Data to version
            metadata: Version metadata
            
        Returns:
            Version ID
        """
        version_id = self._generate_version_id(data)
        
        # Save version data
        version_path = os.path.join(self.versions_path, f"{version_id}.json")
        async with aiofiles.open(version_path, 'w') as f:
            await f.write(json.dumps(data, indent=2))
        
        # Save metadata
        metadata_path = os.path.join(self.metadata_path, f"{version_id}.json")
        async with aiofiles.open(metadata_path, 'w') as f:
            await f.write(json.dumps({
                'version_id': metadata.version_id,
                'timestamp': metadata.timestamp.isoformat(),
                'version_type': metadata.version_type.value,
                'description': metadata.description,
                'author': metadata.author,
                'tags': metadata.tags,
                'properties': metadata.properties
            }, indent=2))
        
        return version_id
    
    async def get_version(self, version_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific version of data.
        
        Args:
            version_id: Version ID to retrieve
            
        Returns:
            Version data or None if not found
        """
        version_path = os.path.join(self.versions_path, f"{version_id}.json")
        
        if not os.path.exists(version_path):
            return None
        
        async with aiofiles.open(version_path, 'r') as f:
            data = await f.read()
            return json.loads(data)
    
    async def get_metadata(self, version_id: str) -> Optional[VersionMetadata]:
        """
        Get metadata for a specific version.
        
        Args:
            version_id: Version ID to retrieve metadata for
            
        Returns:
            Version metadata or None if not found
        """
        metadata_path = os.path.join(self.metadata_path, f"{version_id}.json")
        
        if not os.path.exists(metadata_path):
            return None
        
        async with aiofiles.open(metadata_path, 'r') as f:
            data = await f.read()
            metadata_dict = json.loads(data)
            
            return VersionMetadata(
                version_id=metadata_dict['version_id'],
                timestamp=datetime.fromisoformat(metadata_dict['timestamp']),
                version_type=VersionType(metadata_dict['version_type']),
                description=metadata_dict['description'],
                author=metadata_dict['author'],
                tags=metadata_dict['tags'],
                properties=metadata_dict['properties']
            )
    
    async def list_versions(self,
                          version_type: Optional[VersionType] = None,
                          tags: Optional[List[str]] = None) -> List[str]:
        """
        List available versions.
        
        Args:
            version_type: Filter by version type
            tags: Filter by tags
            
        Returns:
            List of version IDs
        """
        versions = []
        
        for filename in os.listdir(self.metadata_path):
            if not filename.endswith('.json'):
                continue
            
            metadata = await self.get_metadata(filename[:-5])
            if metadata is None:
                continue
            
            if version_type and metadata.version_type != version_type:
                continue
            
            if tags and not all(tag in metadata.tags for tag in tags):
                continue
            
            versions.append(metadata.version_id)
        
        return versions

class LineageTracker:
    """Tracks data lineage and relationships between versions."""
    
    def __init__(self, storage_path: str):
        """
        Initialize the lineage tracker.
        
        Args:
            storage_path: Path to store lineage data
        """
        self.storage_path = storage_path
        self.lineage_path = os.path.join(storage_path, 'lineage')
        
        # Create storage directory
        os.makedirs(self.lineage_path, exist_ok=True)
    
    async def track_lineage(self,
                          version_id: str,
                          parents: List[str],
                          transformations: List[Dict[str, Any]]) -> None:
        """
        Track lineage for a version.
        
        Args:
            version_id: Version ID to track
            parents: List of parent version IDs
            transformations: List of transformations applied
        """
        node = LineageNode(
            version_id=version_id,
            metadata=await VersionStore(self.storage_path).get_metadata(version_id),
            parents=parents,
            children=[],
            transformations=transformations
        )
        
        # Save lineage node
        node_path = os.path.join(self.lineage_path, f"{version_id}.json")
        async with aiofiles.open(node_path, 'w') as f:
            await f.write(json.dumps({
                'version_id': node.version_id,
                'parents': node.parents,
                'children': node.children,
                'transformations': node.transformations
            }, indent=2))
        
        # Update parent nodes
        for parent_id in parents:
            parent_path = os.path.join(self.lineage_path, f"{parent_id}.json")
            if os.path.exists(parent_path):
                async with aiofiles.open(parent_path, 'r') as f:
                    parent_data = json.loads(await f.read())
                
                if version_id not in parent_data['children']:
                    parent_data['children'].append(version_id)
                    
                    async with aiofiles.open(parent_path, 'w') as f:
                        await f.write(json.dumps(parent_data, indent=2))
    
    async def get_lineage(self, version_id: str) -> Optional[LineageNode]:
        """
        Get lineage information for a version.
        
        Args:
            version_id: Version ID to get lineage for
            
        Returns:
            Lineage node or None if not found
        """
        node_path = os.path.join(self.lineage_path, f"{version_id}.json")
        
        if not os.path.exists(node_path):
            return None
        
        async with aiofiles.open(node_path, 'r') as f:
            data = await f.read()
            node_data = json.loads(data)
            
            return LineageNode(
                version_id=node_data['version_id'],
                metadata=await VersionStore(self.storage_path).get_metadata(version_id),
                parents=node_data['parents'],
                children=node_data['children'],
                transformations=node_data['transformations']
            )
    
    async def get_ancestors(self, version_id: str) -> List[LineageNode]:
        """
        Get all ancestors of a version.
        
        Args:
            version_id: Version ID to get ancestors for
            
        Returns:
            List of ancestor lineage nodes
        """
        ancestors = []
        visited = set()
        
        async def _get_ancestors(node_id: str) -> None:
            if node_id in visited:
                return
            
            visited.add(node_id)
            node = await self.get_lineage(node_id)
            
            if node is None:
                return
            
            for parent_id in node.parents:
                parent = await self.get_lineage(parent_id)
                if parent is not None:
                    ancestors.append(parent)
                    await _get_ancestors(parent_id)
        
        await _get_ancestors(version_id)
        return ancestors
    
    async def get_descendants(self, version_id: str) -> List[LineageNode]:
        """
        Get all descendants of a version.
        
        Args:
            version_id: Version ID to get descendants for
            
        Returns:
            List of descendant lineage nodes
        """
        descendants = []
        visited = set()
        
        async def _get_descendants(node_id: str) -> None:
            if node_id in visited:
                return
            
            visited.add(node_id)
            node = await self.get_lineage(node_id)
            
            if node is None:
                return
            
            for child_id in node.children:
                child = await self.get_lineage(child_id)
                if child is not None:
                    descendants.append(child)
                    await _get_descendants(child_id)
        
        await _get_descendants(version_id)
        return descendants

class DataVersioning:
    """Main class for data versioning and lineage tracking."""
    
    def __init__(self, storage_path: str):
        """
        Initialize the data versioning system.
        
        Args:
            storage_path: Path to store versioning data
        """
        self.storage_path = storage_path
        self.version_store = VersionStore(storage_path)
        self.lineage_tracker = LineageTracker(storage_path)
    
    async def create_version(self,
                           data: Dict[str, Any],
                           metadata: VersionMetadata,
                           parents: Optional[List[str]] = None,
                           transformations: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Create a new version of data with lineage tracking.
        
        Args:
            data: Data to version
            metadata: Version metadata
            parents: List of parent version IDs
            transformations: List of transformations applied
            
        Returns:
            Version ID
        """
        # Create version
        version_id = await self.version_store.create_version(data, metadata)
        
        # Track lineage
        await self.lineage_tracker.track_lineage(
            version_id=version_id,
            parents=parents or [],
            transformations=transformations or []
        )
        
        return version_id
    
    async def get_version(self, version_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific version of data.
        
        Args:
            version_id: Version ID to retrieve
            
        Returns:
            Version data or None if not found
        """
        return await self.version_store.get_version(version_id)
    
    async def get_metadata(self, version_id: str) -> Optional[VersionMetadata]:
        """
        Get metadata for a specific version.
        
        Args:
            version_id: Version ID to retrieve metadata for
            
        Returns:
            Version metadata or None if not found
        """
        return await self.version_store.get_metadata(version_id)
    
    async def get_lineage(self, version_id: str) -> Optional[LineageNode]:
        """
        Get lineage information for a version.
        
        Args:
            version_id: Version ID to get lineage for
            
        Returns:
            Lineage node or None if not found
        """
        return await self.lineage_tracker.get_lineage(version_id)
    
    async def get_ancestors(self, version_id: str) -> List[LineageNode]:
        """
        Get all ancestors of a version.
        
        Args:
            version_id: Version ID to get ancestors for
            
        Returns:
            List of ancestor lineage nodes
        """
        return await self.lineage_tracker.get_ancestors(version_id)
    
    async def get_descendants(self, version_id: str) -> List[LineageNode]:
        """
        Get all descendants of a version.
        
        Args:
            version_id: Version ID to get descendants for
            
        Returns:
            List of descendant lineage nodes
        """
        return await self.lineage_tracker.get_descendants(version_id)
    
    async def list_versions(self,
                          version_type: Optional[VersionType] = None,
                          tags: Optional[List[str]] = None) -> List[str]:
        """
        List available versions.
        
        Args:
            version_type: Filter by version type
            tags: Filter by tags
            
        Returns:
            List of version IDs
        """
        return await self.version_store.list_versions(version_type, tags)
