"""
Data lineage tracking and provenance system.
"""

from typing import Dict, List, Optional, Any, Union, Set, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
import uuid
import json
import logging
from pathlib import Path
import networkx as nx
import pandas as pd
import hashlib

logger = logging.getLogger(__name__)

class EntityType(Enum):
    DATASET = "dataset"
    FEATURE = "feature"
    MODEL = "model"
    PIPELINE = "pipeline"
    TRANSFORMATION = "transformation"
    SOURCE = "source"

class OperationType(Enum):
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    TRANSFORM = "transform"
    TRAIN = "train"
    PREDICT = "predict"
    EVALUATE = "evaluate"
    IMPORT = "import"
    EXPORT = "export"

class DataQuality(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNKNOWN = "unknown"

@dataclass
class Entity:
    """Represents a data entity in the lineage graph."""
    entity_id: str
    entity_type: EntityType
    name: str
    description: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    quality: DataQuality = DataQuality.UNKNOWN
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['entity_type'] = self.entity_type.value
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        data['quality'] = self.quality.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Entity':
        """Create from dictionary."""
        data = data.copy()
        data['entity_type'] = EntityType(data['entity_type'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        data['quality'] = DataQuality(data['quality'])
        return cls(**data)

@dataclass
class Operation:
    """Represents an operation in the lineage graph."""
    operation_id: str
    operation_type: OperationType
    entity_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    inputs: List[str] = field(default_factory=list)  # List of input entity IDs
    outputs: List[str] = field(default_factory=list)  # List of output entity IDs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['operation_type'] = self.operation_type.value
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Operation':
        """Create from dictionary."""
        data = data.copy()
        data['operation_type'] = OperationType(data['operation_type'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

class DataLineage:
    """
    Data lineage tracking system that maintains a graph of data entities and operations.
    
    Features:
    - Track data flow between entities
    - Maintain provenance information
    - Query lineage across multiple hops
    - Export/import lineage graphs
    - Generate data quality reports
    """
    
    def __init__(self, storage_backend: str = "memory", **backend_kwargs):
        """
        Initialize the data lineage tracker.
        
        Args:
            storage_backend: Storage backend to use ("memory" or "filesystem")
            **backend_kwargs: Backend-specific configuration
        """
        self.storage_backend = storage_backend
        self.backend_kwargs = backend_kwargs
        self.graph = nx.MultiDiGraph()
        self._init_storage()
    
    def _init_storage(self) -> None:
        """Initialize the storage backend."""
        if self.storage_backend == "filesystem":
            self.storage_path = Path(self.backend_kwargs.get('storage_path', './data/lineage'))
            self.storage_path.mkdir(parents=True, exist_ok=True)
            
            # Load existing graph if exists
            graph_file = self.storage_path / 'lineage_graph.json'
            if graph_file.exists():
                self._load_graph()
    
    def _save_graph(self) -> None:
        """Save the graph to storage."""
        if self.storage_backend == "filesystem":
            graph_file = self.storage_path / 'lineage_graph.json'
            data = {
                'nodes': [
                    {'id': n, 'data': self.graph.nodes[n]}
                    for n in self.graph.nodes()
                ],
                'edges': [
                    {'source': u, 'target': v, 'key': k, 'data': d}
                    for u, v, k, d in self.graph.edges(keys=True, data=True)
                ]
            }
            with open(graph_file, 'w') as f:
                json.dump(data, f, default=str, indent=2)
    
    def _load_graph(self) -> None:
        """Load the graph from storage."""
        if self.storage_backend == "filesystem":
            graph_file = self.storage_path / 'lineage_graph.json'
            if graph_file.exists():
                with open(graph_file, 'r') as f:
                    data = json.load(f)
                
                # Clear existing graph
                self.graph.clear()
                
                # Add nodes
                for node in data['nodes']:
                    self.graph.add_node(node['id'], **node['data'])
                
                # Add edges
                for edge in data['edges']:
                    self.graph.add_edge(
                        edge['source'],
                        edge['target'],
                        key=edge['key'],
                        **edge['data']
                    )
    
    def _generate_id(self, prefix: str = "ent") -> str:
        """Generate a unique ID."""
        return f"{prefix}_{uuid.uuid4().hex}"
    
    def add_entity(
        self,
        entity_type: Union[EntityType, str],
        name: str,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        entity_id: Optional[str] = None,
        quality: Union[DataQuality, str] = DataQuality.UNKNOWN
    ) -> Entity:
        """
        Add or update an entity in the lineage graph.
        
        Args:
            entity_type: Type of the entity
            name: Name of the entity
            description: Description of the entity
            metadata: Additional metadata
            tags: List of tags
            entity_id: Optional custom ID
            quality: Data quality rating
            
        Returns:
            The created or updated entity
        """
        if isinstance(entity_type, str):
            entity_type = EntityType(entity_type.lower())
        if isinstance(quality, str):
            quality = DataQuality(quality.lower())
            
        entity_id = entity_id or self._generate_id(prefix=entity_type.value[:3])
        
        # Check if entity exists
        if entity_id in self.graph.nodes:
            # Update existing entity
            node_data = self.graph.nodes[entity_id]
            entity = Entity.from_dict(node_data['data'])
            entity.name = name
            entity.description = description
            entity.updated_at = datetime.utcnow()
            entity.metadata.update(metadata or {})
            entity.tags = list(set((entity.tags or []) + (tags or [])))
            entity.quality = quality
        else:
            # Create new entity
            entity = Entity(
                entity_id=entity_id,
                entity_type=entity_type,
                name=name,
                description=description,
                metadata=metadata or {},
                tags=tags or [],
                quality=quality
            )
        
        # Add to graph
        self.graph.add_node(
            entity_id,
            type='entity',
            entity_type=entity_type.value,
            data=entity.to_dict(),
            label=name,
            created_at=entity.created_at.isoformat(),
            updated_at=entity.updated_at.isoformat()
        )
        
        self._save_graph()
        return entity
    
    def add_operation(
        self,
        operation_type: Union[OperationType, str],
        entity_id: str,
        inputs: Optional[List[str]] = None,
        outputs: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        operation_id: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> Operation:
        """
        Add an operation to the lineage graph.
        
        Args:
            operation_type: Type of the operation
            entity_id: ID of the entity being operated on
            inputs: List of input entity IDs
            outputs: List of output entity IDs
            parameters: Operation parameters
            metadata: Additional metadata
            operation_id: Optional custom operation ID
            timestamp: Operation timestamp (default: now)
            
        Returns:
            The created operation
        """
        if isinstance(operation_type, str):
            operation_type = OperationType(operation_type.lower())
            
        operation_id = operation_id or self._generate_id(prefix="op")
        timestamp = timestamp or datetime.utcnow()
        
        # Create operation
        operation = Operation(
            operation_id=operation_id,
            operation_type=operation_type,
            entity_id=entity_id,
            timestamp=timestamp,
            parameters=parameters or {},
            metadata=metadata or {},
            inputs=inputs or [],
            outputs=outputs or []
        )
        
        # Add operation node
        self.graph.add_node(
            operation_id,
            type='operation',
            operation_type=operation_type.value,
            data=operation.to_dict(),
            label=f"{operation_type.value.capitalize()}: {entity_id}",
            timestamp=timestamp.isoformat()
        )
        
        # Connect operation to entity
        self.graph.add_edge(
            operation_id,
            entity_id,
            type='produces',
            label='produces'
        )
        
        # Connect inputs to operation
        for input_id in (inputs or []):
            self.graph.add_edge(
                input_id,
                operation_id,
                type='input_to',
                label='input to'
            )
        
        # Connect operation to outputs
        for output_id in (outputs or []):
            self.graph.add_edge(
                operation_id,
                output_id,
                type='produces',
                label='produces'
            )
        
        self._save_graph()
        return operation
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID."""
        if entity_id in self.graph.nodes and self.graph.nodes[entity_id].get('type') == 'entity':
            return Entity.from_dict(self.graph.nodes[entity_id]['data'])
        return None
    
    def get_operation(self, operation_id: str) -> Optional[Operation]:
        """Get an operation by ID."""
        if operation_id in self.graph.nodes and self.graph.nodes[operation_id].get('type') == 'operation':
            return Operation.from_dict(self.graph.nodes[operation_id]['data'])
        return None
    
    def get_lineage(
        self,
        entity_id: str,
        direction: str = 'both',
        max_hops: Optional[int] = None,
        include_operations: bool = True
    ) -> Dict[str, Any]:
        """
        Get the lineage of an entity.
        
        Args:
            entity_id: ID of the entity
            direction: 'upstream', 'downstream', or 'both'
            max_hops: Maximum number of hops to traverse
            include_operations: Whether to include operations in the result
            
        Returns:
            Dictionary containing the lineage graph and metadata
        """
        if direction not in ['upstream', 'downstream', 'both']:
            raise ValueError("direction must be 'upstream', 'downstream', or 'both'")
        
        if entity_id not in self.graph:
            return {'nodes': [], 'edges': []}
        
        # Create a subgraph with the relevant nodes
        nodes = set([entity_id])
        edges = set()
        
        def traverse(node_id: str, current_hops: int, direction: str) -> None:
            if max_hops is not None and current_hops >= max_hops:
                return
                
            if direction in ['upstream', 'both']:
                for pred in self.graph.predecessors(node_id):
                    edge_data = self.graph.get_edge_data(pred, node_id)
                    if edge_data:
                        for key, data in edge_data.items():
                            nodes.add(pred)
                            edges.add((pred, node_id, key, data))
                            traverse(pred, current_hops + 1, direction)
            
            if direction in ['downstream', 'both']:
                for succ in self.graph.successors(node_id):
                    edge_data = self.graph.get_edge_data(node_id, succ)
                    if edge_data:
                        for key, data in edge_data.items():
                            nodes.add(succ)
                            edges.add((node_id, succ, key, data))
                            traverse(succ, current_hops + 1, direction)
        
        traverse(entity_id, 0, direction)
        
        # Filter nodes and edges
        subgraph = {
            'nodes': [
                {
                    'id': n,
                    **self.graph.nodes[n],
                    'data': self._serialize_node_data(self.graph.nodes[n].get('data', {}))
                }
                for n in nodes
                if include_operations or self.graph.nodes[n].get('type') != 'operation'
            ],
            'edges': [
                {
                    'source': u,
                    'target': v,
                    'key': k,
                    **d
                }
                for u, v, k, d in edges
                if include_operations or (
                    self.graph.nodes[u].get('type') != 'operation' and 
                    self.graph.nodes[v].get('type') != 'operation'
                )
            ]
        }
        
        return subgraph
    
    def _serialize_node_data(self, data: Any) -> Any:
        """Recursively serialize node data."""
        if isinstance(data, (str, int, float, bool, type(None))):
            return data
        elif isinstance(data, dict):
            return {k: self._serialize_node_data(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return [self._serialize_node_data(item) for item in data]
        elif hasattr(data, 'isoformat'):  # Handle datetime
            return data.isoformat()
        elif hasattr(data, '__dict__'):
            return self._serialize_node_data(data.__dict__)
        else:
            return str(data)
    
    def get_entity_history(self, entity_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get the history of operations on an entity.
        
        Args:
            entity_id: ID of the entity
            limit: Maximum number of operations to return
            
        Returns:
            List of operations in chronological order
        """
        operations = []
        
        # Find all operations that produced this entity
        for op_id in self.graph.predecessors(entity_id):
            if self.graph.nodes[op_id].get('type') == 'operation':
                operations.append(self.graph.nodes[op_id]['data'])
        
        # Sort by timestamp
        operations.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return operations[:limit]
    
    def export_lineage(self, format: str = 'json') -> str:
        """
        Export the complete lineage graph.
        
        Args:
            format: Export format ('json' or 'graphml')
            
        Returns:
            Serialized lineage data
        """
        if format == 'json':
            data = nx.node_link_data(self.graph)
            return json.dumps(data, indent=2, default=str)
        elif format == 'graphml':
            # Convert to a simpler graph for GraphML export
            G = nx.DiGraph()
            
            for node, data in self.graph.nodes(data=True):
                if data.get('type') == 'entity':
                    G.add_node(node, **{'type': 'entity', 'label': data.get('label', '')})
                else:
                    G.add_node(node, **{'type': 'operation', 'label': data.get('label', '')})
            
            for u, v, data in self.graph.edges(data=True):
                G.add_edge(u, v, **{'type': data.get('type', '')})
            
            return '
'.join(nx.generate_graphml(G))
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @classmethod
    def import_lineage(cls, data: str, format: str = 'json') -> 'DataLineage':
        """
        Import a lineage graph.
        
        Args:
            data: Serialized lineage data
            format: Input format ('json' or 'graphml')
            
        Returns:
            New DataLineage instance
        """
        lineage = cls()
        
        if format == 'json':
            graph_data = json.loads(data)
            lineage.graph = nx.node_link_graph(graph_data)
        elif format == 'graphml':
            # Create a simple graph from GraphML
            G = nx.parse_graphml(data)
            
            # Convert to MultiDiGraph and add necessary attributes
            lineage.graph = nx.MultiDiGraph()
            
            for node, data in G.nodes(data=True):
                if data.get('type') == 'entity':
                    lineage.graph.add_node(node, type='entity', label=data.get('label', ''))
                else:
                    lineage.graph.add_node(node, type='operation', label=data.get('label', ''))
            
            for u, v, data in G.edges(data=True):
                lineage.graph.add_edge(u, v, **{'type': data.get('type', '')})
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return lineage
    
    def generate_report(self, entity_id: str) -> Dict[str, Any]:
        """
        Generate a data quality and lineage report for an entity.
        
        Args:
            entity_id: ID of the entity
            
        Returns:
            Dictionary containing the report
        """
        entity = self.get_entity(entity_id)
        if not entity:
            return {'error': f'Entity {entity_id} not found'}
        
        # Get lineage
        lineage = self.get_lineage(entity_id, direction='both', max_hops=3)
        
        # Count entities and operations
        entity_count = sum(1 for n in lineage['nodes'] if n.get('type') == 'entity')
        operation_count = sum(1 for n in lineage['nodes'] if n.get('type') == 'operation')
        
        # Get quality metrics
        quality_metrics = {
            'entity_quality': entity.quality.value,
            'upstream_quality': self._calculate_upstream_quality(entity_id),
            'completeness': self._calculate_completeness(entity_id),
            'freshness': self._calculate_freshness(entity_id)
        }
        
        # Get recent operations
        recent_operations = self.get_entity_history(entity_id, limit=5)
        
        return {
            'entity': entity.to_dict(),
            'summary': {
                'entity_count': entity_count,
                'operation_count': operation_count,
                'edge_count': len(lineage['edges']),
                'quality_metrics': quality_metrics
            },
            'recent_operations': recent_operations,
            'lineage_summary': {
                'upstream': self._summarize_lineage(entity_id, 'upstream'),
                'downstream': self._summarize_lineage(entity_id, 'downstream')
            }
        }
    
    def _calculate_upstream_quality(self, entity_id: str) -> Dict[str, Any]:
        """Calculate quality metrics for upstream entities."""
        upstream = self.get_lineage(entity_id, direction='upstream', max_hops=1)
        qualities = []
        
        for node in upstream['nodes']:
            if node.get('type') == 'entity' and 'data' in node:
                try:
                    entity = Entity.from_dict(node['data'])
                    qualities.append(entity.quality.value)
                except:
                    pass
        
        if not qualities:
            return {'average_quality': 'unknown', 'worst_quality': 'unknown'}
        
        # Simple quality scoring (could be more sophisticated)
        quality_scores = {
            'excellent': 4,
            'good': 3,
            'fair': 2,
            'poor': 1,
            'unknown': 0
        }
        
        scores = [quality_scores.get(q.lower(), 0) for q in qualities]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        # Convert back to quality rating
        quality_map = {v: k for k, v in quality_scores.items()}
        closest = min(quality_scores.values(), key=lambda x: abs(x - avg_score))
        
        return {
            'average_quality': quality_map.get(round(avg_score), 'unknown'),
            'worst_quality': quality_map.get(min(scores), 'unknown') if scores else 'unknown',
            'upstream_count': len(qualities)
        }
    
    def _calculate_completeness(self, entity_id: str) -> Dict[str, Any]:
        """Calculate completeness metrics for an entity."""
        # This is a simplified example - in practice, you'd check actual data
        entity = self.get_entity(entity_id)
        if not entity:
            return {'score': 0, 'missing_fields': []}
        
        # Check for common required fields based on entity type
        required_fields = {
            EntityType.DATASET: ['name', 'description', 'schema'],
            EntityType.FEATURE: ['name', 'data_type', 'source'],
            EntityType.MODEL: ['name', 'version', 'framework'],
            EntityType.PIPELINE: ['name', 'steps'],
            EntityType.TRANSFORMATION: ['name', 'inputs', 'outputs'],
            EntityType.SOURCE: ['name', 'connection_details']
        }.get(entity.entity_type, [])
        
        missing = []
        for field in required_fields:
            if not getattr(entity, field, None) and field not in entity.metadata:
                missing.append(field)
        
        score = 1.0 - (len(missing) / len(required_fields)) if required_fields else 1.0
        
        return {
            'score': score,
            'missing_fields': missing,
            'has_required_fields': len(missing) == 0
        }
    
    def _calculate_freshness(self, entity_id: str) -> Dict[str, Any]:
        """Calculate freshness metrics for an entity."""
        operations = self.get_entity_history(entity_id, limit=1)
        if not operations:
            return {'last_updated': None, 'days_since_update': None, 'freshness': 'unknown'}
        
        last_op = operations[0]
        last_updated = datetime.fromisoformat(last_op['timestamp'])
        days_since_update = (datetime.utcnow() - last_updated).days
        
        # Simple freshness rating
        if days_since_update < 1:
            freshness = 'fresh'
        elif days_since_update < 7:
            freshness = 'recent'
        elif days_since_update < 30:
            freshness = 'stale'
        else:
            freshness = 'outdated'
        
        return {
            'last_updated': last_updated.isoformat(),
            'days_since_update': days_since_update,
            'freshness': freshness
        }
    
    def _summarize_lineage(self, entity_id: str, direction: str) -> Dict[str, Any]:
        """Generate a summary of the lineage in a specific direction."""
        lineage = self.get_lineage(entity_id, direction=direction, max_hops=2)
        
        entities = []
        for node in lineage['nodes']:
            if node.get('type') == 'entity' and node.get('id') != entity_id:
                entities.append({
                    'id': node.get('id'),
                    'type': node.get('entity_type'),
                    'name': node.get('label'),
                    'quality': node.get('data', {}).get('quality', 'unknown')
                })
        
        return {
            'entity_count': len(entities),
            'entity_types': {e['type']: sum(1 for x in entities if x['type'] == e['type']) for e in entities},
            'entities': entities[:10]  # Limit to first 10 for summary
        }
