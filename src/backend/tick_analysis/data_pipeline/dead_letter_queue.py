"""
Dead Letter Queue Module

This module implements a dead letter queue for handling failed data processing
batches with retry mechanisms and error tracking.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json
import os

logger = logging.getLogger(__name__)

@dataclass
class FailedBatch:
    """Represents a failed batch of data with metadata."""
    batch: List[Dict[str, Any]]
    error: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    retry_count: int = 0
    max_retries: int = 3
    last_retry: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class DeadLetterQueue:
    """Handles failed data processing batches with retry mechanisms."""
    
    def __init__(self, 
                 storage_path: str = './data/dlq',
                 max_retries: int = 3,
                 retry_delay: int = 300,  # 5 minutes
                 max_queue_size: int = 10000):
        """
        Initialize the dead letter queue.
        
        Args:
            storage_path: Path to store failed batches
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            max_queue_size: Maximum number of items in memory queue
        """
        self.storage_path = storage_path
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_queue_size = max_queue_size
        
        # Create storage directory if it doesn't exist
        os.makedirs(storage_path, exist_ok=True)
        
        # In-memory queue for recent failures
        self.queue = asyncio.Queue(maxsize=max_queue_size)
        
        # Metrics
        self.metrics = {
            'total_failures': 0,
            'retried': 0,
            'permanent_failures': 0,
            'current_queue_size': 0
        }
        
        # Start processing task
        self._processing_task = None
        self._running = False
    
    async def start(self) -> None:
        """Start the dead letter queue processor."""
        if self._running:
            return
            
        self._running = True
        self._processing_task = asyncio.create_task(self._process_queue())
        logger.info("Dead letter queue processor started")
    
    async def stop(self) -> None:
        """Stop the dead letter queue processor."""
        if not self._running:
            return
            
        self._running = False
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
        logger.info("Dead letter queue processor stopped")
    
    async def add_failed_batch(self, 
                             batch: List[Dict[str, Any]], 
                             error: Exception,
                             metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a failed batch to the queue.
        
        Args:
            batch: The failed batch of data
            error: The exception that caused the failure
            metadata: Additional metadata about the failure
        """
        failed_batch = FailedBatch(
            batch=batch,
            error=str(error),
            max_retries=self.max_retries,
            metadata=metadata or {}
        )
        
        try:
            await self.queue.put(failed_batch)
            self.metrics['total_failures'] += 1
            self.metrics['current_queue_size'] = self.queue.qsize()
            
            # Persist to disk
            await self._persist_failed_batch(failed_batch)
            
        except asyncio.QueueFull:
            logger.error("Dead letter queue is full, failed batch will be lost")
            self.metrics['permanent_failures'] += 1
    
    async def _process_queue(self) -> None:
        """Process the dead letter queue."""
        while self._running:
            try:
                # Get next failed batch
                failed_batch = await self.queue.get()
                
                # Check if we should retry
                if failed_batch.retry_count < failed_batch.max_retries:
                    # Check if enough time has passed since last retry
                    if (failed_batch.last_retry is None or 
                        datetime.utcnow() - failed_batch.last_retry > timedelta(seconds=self.retry_delay)):
                        
                        # Retry the batch
                        await self._retry_batch(failed_batch)
                        failed_batch.retry_count += 1
                        failed_batch.last_retry = datetime.utcnow()
                        
                        # Update metrics
                        self.metrics['retried'] += 1
                        
                        # Put back in queue if not at max retries
                        if failed_batch.retry_count < failed_batch.max_retries:
                            await self.queue.put(failed_batch)
                        else:
                            self.metrics['permanent_failures'] += 1
                            await self._archive_failed_batch(failed_batch)
                    else:
                        # Not enough time has passed, put back in queue
                        await self.queue.put(failed_batch)
                else:
                    # Max retries reached, archive the batch
                    self.metrics['permanent_failures'] += 1
                    await self._archive_failed_batch(failed_batch)
                
                self.queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing dead letter queue: {e}")
                await asyncio.sleep(1)
    
    async def _retry_batch(self, failed_batch: FailedBatch) -> None:
        """
        Retry processing a failed batch.
        
        Args:
            failed_batch: The failed batch to retry
        """
        try:
            # Here you would implement the actual retry logic
            # This could involve calling back to the pipeline or a specific processor
            logger.info(f"Retrying batch (attempt {failed_batch.retry_count + 1}/{failed_batch.max_retries})")
            
            # Update the persisted batch with new retry information
            await self._persist_failed_batch(failed_batch)
            
        except Exception as e:
            logger.error(f"Error retrying batch: {e}")
    
    async def _persist_failed_batch(self, failed_batch: FailedBatch) -> None:
        """
        Persist a failed batch to disk.
        
        Args:
            failed_batch: The failed batch to persist
        """
        try:
            # Create a unique filename
            timestamp = failed_batch.timestamp.strftime('%Y%m%d_%H%M%S_%f')
            filename = f"failed_batch_{timestamp}.json"
            filepath = os.path.join(self.storage_path, filename)
            
            # Convert to serializable format
            data = {
                'batch': failed_batch.batch,
                'error': failed_batch.error,
                'timestamp': failed_batch.timestamp.isoformat(),
                'retry_count': failed_batch.retry_count,
                'max_retries': failed_batch.max_retries,
                'last_retry': failed_batch.last_retry.isoformat() if failed_batch.last_retry else None,
                'metadata': failed_batch.metadata
            }
            
            # Write to file
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error persisting failed batch: {e}")
    
    async def _archive_failed_batch(self, failed_batch: FailedBatch) -> None:
        """
        Archive a permanently failed batch.
        
        Args:
            failed_batch: The failed batch to archive
        """
        try:
            # Move to archive directory
            archive_dir = os.path.join(self.storage_path, 'archive')
            os.makedirs(archive_dir, exist_ok=True)
            
            timestamp = failed_batch.timestamp.strftime('%Y%m%d_%H%M%S_%f')
            filename = f"archived_batch_{timestamp}.json"
            filepath = os.path.join(archive_dir, filename)
            
            # Convert to serializable format
            data = {
                'batch': failed_batch.batch,
                'error': failed_batch.error,
                'timestamp': failed_batch.timestamp.isoformat(),
                'retry_count': failed_batch.retry_count,
                'max_retries': failed_batch.max_retries,
                'last_retry': failed_batch.last_retry.isoformat() if failed_batch.last_retry else None,
                'metadata': failed_batch.metadata,
                'archived_at': datetime.utcnow().isoformat()
            }
            
            # Write to archive file
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error archiving failed batch: {e}")
    
    def get_metrics(self) -> Dict[str, int]:
        """Get current queue metrics."""
        return self.metrics.copy() 