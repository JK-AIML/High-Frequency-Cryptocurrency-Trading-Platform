"""
Data Pipeline Module

This module provides a comprehensive data processing pipeline for tick data,
including data ingestion, validation, transformation, and storage.
"""

import logging
import asyncio
import time
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import json
import os
from .sources import DataSource, FileSource, WebSocketSource, KafkaSource
from .processors import DataProcessor, DataValidator, DataTransformer
from .storage import EnhancedInfluxDBStorage, EnhancedParquetStorage
from .dead_letter_queue import DeadLetterQueue
from .recovery import PipelineRecovery
from .monitoring import PipelineMonitor
from .validation import EnhancedValidationProcessor, ValidationResult

logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """Configuration for the data pipeline."""
    source_type: str  # 'file', 'websocket', 'kafka'
    source_config: Dict[str, Any]
    processors: List[Dict[str, Any]]
    storage_type: str  # 'influxdb', 'parquet'
    storage_config: Dict[str, Any]
    batch_size: int = 1000
    max_retries: int = 3
    retry_delay: float = 1.0
    metrics_port: int = 9090
    recovery_enabled: bool = True
    monitoring_enabled: bool = True
    dlq_enabled: bool = True
    quality_monitoring_enabled: bool = True

class DataPipeline:
    """Main data pipeline class."""
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the data pipeline.
        Ensures comprehensive data validation is always performed as the first step.
        """
        self.config = config
        self.pipeline_id = f"pipeline_{int(time.time())}"
        
        # Initialize components
        self._init_components()
        
        # Pipeline state
        self._running = False
        self._processing_task = None
        self._metrics_task = None
        self._start_time = None
        
        # Metrics
        self.metrics = {
            'processed_records': 0,
            'processing_errors': {},
            'processing_latency': 0.0,
            'batch_size': 0,
            'queue_size': 0
        }
    
    def _init_components(self) -> None:
        """Initialize pipeline components."""
        # Initialize data source
        if self.config.source_type == 'file':
            self.source = FileSource(**self.config.source_config)
        elif self.config.source_type == 'websocket':
            self.source = WebSocketSource(**self.config.source_config)
        elif self.config.source_type == 'kafka':
            self.source = KafkaSource(**self.config.source_config)
        else:
            raise ValueError(f"Unsupported source type: {self.config.source_type}")
        
        # Always add enhanced validation as the first processor
        validation_config = self.config.processors[0]['config'] if self.config.processors and self.config.processors[0]['type'] == 'validation' else {}
        self.validation_processor = EnhancedValidationProcessor(**validation_config)
        self.processors: List[DataProcessor] = [self.validation_processor]
        for proc_config in self.config.processors:
            if proc_config['type'] == 'validation':
                continue  # Already added
            elif proc_config['type'] == 'transformation':
                self.processors.append(TransformationProcessor(**proc_config['config']))
            else:
                raise ValueError(f"Unsupported processor type: {proc_config['type']}")
        
        # Initialize storage
        if self.config.storage_type == 'influxdb':
            self.storage = EnhancedInfluxDBStorage(**self.config.storage_config)
        elif self.config.storage_type == 'parquet':
            self.storage = EnhancedParquetStorage(**self.config.storage_config)
        else:
            raise ValueError(f"Unsupported storage type: {self.config.storage_type}")
        
        # Initialize Dead Letter Queue if enabled
        if self.config.dlq_enabled:
            self.dlq = DeadLetterQueue(
                pipeline_id=self.pipeline_id,
                max_retries=self.config.max_retries,
                retry_delay=self.config.retry_delay
            )
        else:
            self.dlq = None
        
        # Initialize Pipeline Recovery if enabled
        if self.config.recovery_enabled:
            self.recovery = PipelineRecovery(
                pipeline_id=self.pipeline_id,
                storage_path='./data/recovery'
            )
        else:
            self.recovery = None
        
        # Initialize Pipeline Monitor if enabled
        if self.config.monitoring_enabled:
            self.monitor = PipelineMonitor(
                pipeline_id=self.pipeline_id,
                metrics_port=self.config.metrics_port
            )
        else:
            self.monitor = None
    
    async def start(self) -> None:
        """Start the data pipeline."""
        if self._running:
            return
        
        try:
            # Start components
            await self.source.start()
            await self.storage.start()
            
            if self.dlq:
                await self.dlq.start()
            
            if self.recovery:
                await self.recovery.start()
            
            if self.monitor:
                await self.monitor.start()
            
            # Start processing
            self._running = True
            self._start_time = time.time()
            self._processing_task = asyncio.create_task(self._processing_loop())
            self._metrics_task = asyncio.create_task(self._metrics_loop())
            
            logger.info(f"Pipeline {self.pipeline_id} started successfully")
            
        except Exception as e:
            logger.error(f"Error starting pipeline: {e}")
            await self.stop()
            raise
    
    async def stop(self) -> None:
        """Stop the data pipeline."""
        if not self._running:
            return
        
        self._running = False
        
        # Stop processing tasks
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
        
        if self._metrics_task:
            self._metrics_task.cancel()
            try:
                await self._metrics_task
            except asyncio.CancelledError:
                pass
        
        # Stop components
        try:
            await self.source.stop()
            await self.storage.stop()
            
            if self.dlq:
                await self.dlq.stop()
            
            if self.recovery:
                await self.recovery.stop()
            
            if self.monitor:
                await self.monitor.stop()
            
        except Exception as e:
            logger.error(f"Error stopping pipeline: {e}")
        
        logger.info(f"Pipeline {self.pipeline_id} stopped")
    
    async def _processing_loop(self) -> None:
        """Main processing loop."""
        batch = []
        last_checkpoint = time.time()
        
        while self._running:
            try:
                # Read data from source
                data = await self.source.read()
                if data is None:
                    await asyncio.sleep(0.1)
                    continue
                
                # Add to batch
                batch.append(data)
                
                # Process batch if full
                if len(batch) >= self.config.batch_size:
                    await self._process_batch(batch)
                    batch = []
                    
                    # Update checkpoint
                    if self.recovery:
                        await self.recovery.update_state(
                            last_processed_record=data,
                            metrics=self.metrics
                        )
                    last_checkpoint = time.time()
                
                # Force checkpoint every 5 minutes
                if time.time() - last_checkpoint > 300 and batch:
                    await self._process_batch(batch)
                    batch = []
                    
                    if self.recovery:
                        await self.recovery.update_state(
                            last_processed_record=data,
                            metrics=self.metrics
                        )
                    last_checkpoint = time.time()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                if self.dlq:
                    await self.dlq.add_failed_batch(batch, str(e))
                batch = []
                await asyncio.sleep(1)
    
    async def _process_batch(self, batch: List[Dict[str, Any]]) -> None:
        """
        Process a batch of records.
        
        Args:
            batch: List of records to process
        """
        start_time = time.time()
        
        try:
            # Update metrics
            self.metrics['batch_size'] = len(batch)
            self.metrics['queue_size'] = len(batch)
            
            # Process records
            processed_batch = batch
            for processor in self.processors:
                processed_batch = await processor.process(processed_batch)
            
            # Store processed data
            await self.storage.store(processed_batch)
            
            # Update pipeline metrics
            self.metrics['processed_records'] += len(batch)
            self.metrics['processing_latency'] = time.time() - start_time
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            self.metrics['processing_errors'][str(e)] = self.metrics['processing_errors'].get(str(e), 0) + 1
            
            if self.dlq:
                await self.dlq.add_failed_batch(batch, str(e))
            
            raise
    
    async def _metrics_loop(self) -> None:
        """Metrics collection loop."""
        while self._running:
            try:
                if self.monitor:
                    await self.monitor.record_metrics(self.metrics)
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics loop: {e}")
                await asyncio.sleep(1)
    
    async def recover(self) -> None:
        """Recover the pipeline from the last checkpoint."""
        if not self.recovery:
            logger.warning("Recovery not enabled")
            return
        
        try:
            # Load last checkpoint
            state = await self.recovery.load_checkpoint()
            if not state:
                logger.warning("No checkpoint found")
                return
            
            # Update metrics
            self.metrics = state.metrics
            
            # Resume processing from last record
            if state.last_processed_record:
                await self.source.seek(state.last_processed_record)
            
            logger.info(f"Pipeline recovered from checkpoint at {state.timestamp}")
            
        except Exception as e:
            logger.error(f"Error recovering pipeline: {e}")
            raise
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get current pipeline information."""
        info = {
            'pipeline_id': self.pipeline_id,
            'status': 'running' if self._running else 'stopped',
            'uptime': time.time() - self._start_time if self._start_time else 0,
            'metrics': self.metrics
        }
        
        if self.monitor:
            info['monitoring'] = self.monitor.get_monitoring_info()
        
        return info

    async def process_record(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a single record through the pipeline, with validation and error handling.
        Logs and handles validation errors/warnings.
        """
        # Validate first
        validation_result: ValidationResult = await self.validation_processor.validate(record)
        if not validation_result.is_valid:
            # Log errors, optionally send to monitoring/alerting
            for err in validation_result.errors:
                logger.error(f"Validation error: {err['message']} (path: {err['path']})")
            # Optionally, send to dead letter queue or monitoring
            return None  # Drop invalid record
        # Continue with other processors
        data = record
        for processor in self.processors[1:]:
            data = await processor.process(data)
        return data

# Example usage
async def example_pipeline():
    """Example of how to use the data pipeline."""
    config = PipelineConfig(
        source_type='websocket',
        source_config={
            'url': 'wss://ws-feed.example.com',
            'subscriptions': [
                {'type': 'subscribe', 'product_ids': ['BTC-USD'], 'channels': ['ticker']}
            ]
        },
        storage_type='parquet',
        storage_config={
            'base_path': './data/ticks',
            'partition_cols': ['symbol', 'date']
        },
        batch_size=1000,
        flush_interval=1.0
    )
    
    pipeline = DataPipeline(config)
    
    try:
        # Start the pipeline
        await pipeline.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
    finally:
        await pipeline.stop()

if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(example_pipeline())
