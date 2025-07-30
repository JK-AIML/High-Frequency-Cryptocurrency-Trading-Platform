"""
Pipeline Recovery Module

This module provides functionality for pipeline state management, checkpointing,
and recovery mechanisms to ensure data processing continuity.
"""

import logging
import json
import os
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
import aiofiles
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class PipelineState:
    """Represents the state of the pipeline at a checkpoint."""
    pipeline_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    last_processed_record: Optional[Dict[str, Any]] = None
    last_processed_timestamp: Optional[datetime] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    source_position: Dict[str, Any] = field(default_factory=dict)
    batch_state: Dict[str, Any] = field(default_factory=dict)
    error_count: int = 0
    processed_count: int = 0
    checkpoint_hash: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        state_dict = asdict(self)
        state_dict['timestamp'] = self.timestamp.isoformat()
        if self.last_processed_timestamp:
            state_dict['last_processed_timestamp'] = self.last_processed_timestamp.isoformat()
        return state_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineState':
        """Create state from dictionary."""
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if data.get('last_processed_timestamp'):
            data['last_processed_timestamp'] = datetime.fromisoformat(data['last_processed_timestamp'])
        return cls(**data)

class PipelineRecovery:
    """Manages pipeline state and recovery mechanisms."""
    
    def __init__(self, 
                 pipeline_id: str,
                 storage_path: str = './data/checkpoints',
                 checkpoint_interval: int = 60,  # seconds
                 max_checkpoints: int = 10):
        """
        Initialize pipeline recovery system.
        
        Args:
            pipeline_id: Unique identifier for the pipeline
            storage_path: Path to store checkpoints
            checkpoint_interval: Interval between checkpoints in seconds
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self.pipeline_id = pipeline_id
        self.storage_path = storage_path
        self.checkpoint_interval = checkpoint_interval
        self.max_checkpoints = max_checkpoints
        
        # Create storage directory
        os.makedirs(storage_path, exist_ok=True)
        
        # Current state
        self.current_state = PipelineState(pipeline_id=pipeline_id)
        
        # Recovery state
        self._recovering = False
        self._checkpoint_task = None
        self._last_checkpoint = None
    
    async def start(self) -> None:
        """Start the recovery system."""
        if self._checkpoint_task is not None:
            return
            
        # Try to load last checkpoint
        await self.load_last_checkpoint()
        
        # Start checkpoint task
        self._checkpoint_task = asyncio.create_task(self._checkpoint_loop())
        logger.info(f"Pipeline recovery system started for pipeline {self.pipeline_id}")
    
    async def stop(self) -> None:
        """Stop the recovery system."""
        if self._checkpoint_task is None:
            return
            
        self._checkpoint_task.cancel()
        try:
            await self._checkpoint_task
        except asyncio.CancelledError:
            pass
            
        # Save final checkpoint
        await self.save_checkpoint()
        logger.info(f"Pipeline recovery system stopped for pipeline {self.pipeline_id}")
    
    async def update_state(self, 
                          last_record: Optional[Dict[str, Any]] = None,
                          metrics: Optional[Dict[str, Any]] = None,
                          source_position: Optional[Dict[str, Any]] = None,
                          batch_state: Optional[Dict[str, Any]] = None,
                          error_count: Optional[int] = None,
                          processed_count: Optional[int] = None) -> None:
        """
        Update the current pipeline state.
        
        Args:
            last_record: Last processed record
            metrics: Current pipeline metrics
            source_position: Current position in the data source
            batch_state: Current batch processing state
            error_count: Current error count
            processed_count: Current processed record count
        """
        if last_record:
            self.current_state.last_processed_record = last_record
            self.current_state.last_processed_timestamp = datetime.utcnow()
        
        if metrics:
            self.current_state.metrics.update(metrics)
        
        if source_position:
            self.current_state.source_position.update(source_position)
        
        if batch_state:
            self.current_state.batch_state.update(batch_state)
        
        if error_count is not None:
            self.current_state.error_count = error_count
        
        if processed_count is not None:
            self.current_state.processed_count = processed_count
    
    async def save_checkpoint(self) -> None:
        """Save current state as a checkpoint."""
        try:
            # Update timestamp
            self.current_state.timestamp = datetime.utcnow()
            
            # Calculate state hash
            state_dict = self.current_state.to_dict()
            state_json = json.dumps(state_dict, sort_keys=True)
            state_hash = hashlib.sha256(state_json.encode()).hexdigest()
            self.current_state.checkpoint_hash = state_hash
            
            # Save checkpoint
            filename = f"checkpoint_{self.pipeline_id}_{self.current_state.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(self.storage_path, filename)
            
            async with aiofiles.open(filepath, 'w') as f:
                await f.write(state_json)
            
            self._last_checkpoint = self.current_state.timestamp
            logger.info(f"Checkpoint saved: {filename}")
            
            # Clean old checkpoints
            await self._clean_old_checkpoints()
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
            raise
    
    async def load_last_checkpoint(self) -> Optional[PipelineState]:
        """
        Load the most recent checkpoint.
        
        Returns:
            The loaded pipeline state or None if no checkpoint exists
        """
        try:
            # List checkpoint files
            checkpoints = []
            for filename in os.listdir(self.storage_path):
                if filename.startswith(f"checkpoint_{self.pipeline_id}_"):
                    filepath = os.path.join(self.storage_path, filename)
                    checkpoints.append((filepath, os.path.getmtime(filepath)))
            
            if not checkpoints:
                return None
            
            # Get most recent checkpoint
            latest_checkpoint = max(checkpoints, key=lambda x: x[1])[0]
            
            # Load checkpoint
            async with aiofiles.open(latest_checkpoint, 'r') as f:
                state_json = await f.read()
            
            state_dict = json.loads(state_json)
            self.current_state = PipelineState.from_dict(state_dict)
            
            logger.info(f"Loaded checkpoint: {os.path.basename(latest_checkpoint)}")
            return self.current_state
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return None
    
    async def _checkpoint_loop(self) -> None:
        """Periodically save checkpoints."""
        while True:
            try:
                await asyncio.sleep(self.checkpoint_interval)
                await self.save_checkpoint()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in checkpoint loop: {e}")
                await asyncio.sleep(1)
    
    async def _clean_old_checkpoints(self) -> None:
        """Remove old checkpoints keeping only the most recent ones."""
        try:
            # List checkpoint files
            checkpoints = []
            for filename in os.listdir(self.storage_path):
                if filename.startswith(f"checkpoint_{self.pipeline_id}_"):
                    filepath = os.path.join(self.storage_path, filename)
                    checkpoints.append((filepath, os.path.getmtime(filepath)))
            
            # Sort by modification time
            checkpoints.sort(key=lambda x: x[1], reverse=True)
            
            # Remove old checkpoints
            for filepath, _ in checkpoints[self.max_checkpoints:]:
                try:
                    os.remove(filepath)
                    logger.info(f"Removed old checkpoint: {os.path.basename(filepath)}")
                except Exception as e:
                    logger.error(f"Error removing old checkpoint {filepath}: {e}")
                    
        except Exception as e:
            logger.error(f"Error cleaning old checkpoints: {e}")
    
    def get_recovery_info(self) -> Dict[str, Any]:
        """Get information about the recovery state."""
        return {
            'pipeline_id': self.pipeline_id,
            'recovering': self._recovering,
            'last_checkpoint': self._last_checkpoint.isoformat() if self._last_checkpoint else None,
            'current_state': self.current_state.to_dict()
        } 