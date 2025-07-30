"""
Data Processing Module

This module provides stream processing capabilities and feature store
implementation for real-time data analysis.
"""

import logging
import asyncio
import json
from typing import Dict, List, Any, Optional, Union, Callable, Awaitable, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
import joblib
import os
from collections import defaultdict
import polars as pl
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

@dataclass
class StreamConfig:
    """Configuration for stream processing."""
    window_size: int  # Number of records in sliding window
    slide_interval: int  # Number of records to slide window
    watermark_delay: int  # Maximum delay for late data
    allowed_lateness: int  # Maximum lateness for late data
    state_backend: str = "memory"  # State backend type
    checkpoint_interval: int = 1000  # Checkpoint interval in records

@dataclass
class FeatureConfig:
    """Configuration for feature store."""
    name: str
    description: str
    data_type: str
    tags: List[str]
    version: str = "1.0"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

@dataclass
class ProcessingConfig:
    """Configuration for data processing."""
    batch_size: int = 1000
    num_threads: int = 4
    window_size: int = 100
    min_periods: int = 20
    processing_timeout: int = 30

class StreamProcessor:
    """Stream processing with windowing and state management."""
    
    def __init__(self, config: StreamConfig):
        """
        Initialize stream processor.
        
        Args:
            config: Stream processing configuration
        """
        self.config = config
        self.windows = defaultdict(list)
        self.state = {}
        self.checkpoints = {}
        self._handlers = []
        self._processing = False
    
    async def process_record(self, record: Dict[str, Any], key: str) -> None:
        """
        Process a single record.
        
        Args:
            record: Data record to process
            key: Record key for windowing
        """
        # Add record to window
        self.windows[key].append(record)
        
        # Check if window is full
        if len(self.windows[key]) >= self.config.window_size:
            # Process window
            await self._process_window(key)
            
            # Slide window
            self.windows[key] = self.windows[key][self.config.slide_interval:]
        
        # Update state
        self._update_state(record, key)
        
        # Checkpoint if needed
        if len(self.state) % self.config.checkpoint_interval == 0:
            await self._create_checkpoint()
    
    async def _process_window(self, key: str) -> None:
        """
        Process a window of records.
        
        Args:
            key: Window key
        """
        window_data = self.windows[key]
        
        # Call handlers
        for handler in self._handlers:
            try:
                await handler(window_data, key)
            except Exception as e:
                logger.error(f"Error in window handler: {e}")
    
    def _update_state(self, record: Dict[str, Any], key: str) -> None:
        """
        Update processing state.
        
        Args:
            record: Data record
            key: Record key
        """
        if key not in self.state:
            self.state[key] = {
                'count': 0,
                'last_updated': datetime.utcnow(),
                'metrics': defaultdict(float)
            }
        
        state = self.state[key]
        state['count'] += 1
        state['last_updated'] = datetime.utcnow()
        
        # Update metrics
        for field, value in record.items():
            if isinstance(value, (int, float)):
                state['metrics'][field] = value
    
    async def _create_checkpoint(self) -> None:
        """Create processing checkpoint."""
        checkpoint = {
            'timestamp': datetime.utcnow(),
            'state': dict(self.state),
            'windows': {k: len(v) for k, v in self.windows.items()}
        }
        
        checkpoint_id = f"checkpoint_{len(self.checkpoints)}"
        self.checkpoints[checkpoint_id] = checkpoint
        
        # Keep only last 5 checkpoints
        if len(self.checkpoints) > 5:
            oldest = min(self.checkpoints.keys())
            del self.checkpoints[oldest]
    
    async def restore_checkpoint(self, checkpoint_id: str) -> None:
        """
        Restore processing state from checkpoint.
        
        Args:
            checkpoint_id: Checkpoint identifier
        """
        if checkpoint_id not in self.checkpoints:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")
        
        checkpoint = self.checkpoints[checkpoint_id]
        self.state = checkpoint['state']
        self.windows = defaultdict(list)
    
    def add_handler(self, handler: Callable[[List[Dict[str, Any]], str], Awaitable[None]]) -> None:
        """
        Add window processing handler.
        
        Args:
            handler: Async function to handle window data
        """
        self._handlers.append(handler)
    
    def get_state(self, key: Optional[str] = None) -> Dict[str, Any]:
        """
        Get processing state.
        
        Args:
            key: Optional state key
            
        Returns:
            State data
        """
        if key:
            return self.state.get(key, {})
        return dict(self.state)
    
    def get_window_size(self, key: str) -> int:
        """
        Get current window size.
        
        Args:
            key: Window key
            
        Returns:
            Number of records in window
        """
        return len(self.windows[key])

class FeatureStore:
    """Feature store for managing and serving features."""
    
    def __init__(self, base_path: str):
        """
        Initialize feature store.
        
        Args:
            base_path: Base path for feature storage
        """
        self.base_path = base_path
        self.features = {}
        self.scalers = {}
        self._ensure_directory_exists()
    
    def _ensure_directory_exists(self) -> None:
        """Ensure feature store directory exists."""
        os.makedirs(self.base_path, exist_ok=True)
    
    def register_feature(self, config: FeatureConfig) -> None:
        """
        Register a new feature.
        
        Args:
            config: Feature configuration
        """
        if config.name in self.features:
            raise ValueError(f"Feature {config.name} already registered")
        
        config.created_at = datetime.utcnow()
        config.updated_at = config.created_at
        
        self.features[config.name] = config
        
        # Create feature directory
        feature_path = os.path.join(self.base_path, config.name)
        os.makedirs(feature_path, exist_ok=True)
        
        # Save feature metadata
        self._save_feature_metadata(config)
    
    def _save_feature_metadata(self, config: FeatureConfig) -> None:
        """
        Save feature metadata.
        
        Args:
            config: Feature configuration
        """
        metadata_path = os.path.join(self.base_path, config.name, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump({
                'name': config.name,
                'description': config.description,
                'data_type': config.data_type,
                'tags': config.tags,
                'version': config.version,
                'created_at': config.created_at.isoformat(),
                'updated_at': config.updated_at.isoformat()
            }, f, indent=2)
    
    async def store_feature(self, name: str, data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> None:
        """
        Store feature data.
        
        Args:
            name: Feature name
            data: Feature data
        """
        if name not in self.features:
            raise ValueError(f"Feature {name} not registered")
        
        feature_path = os.path.join(self.base_path, name)
        
        # Convert to DataFrame
        if not isinstance(data, list):
            data = [data]
        df = pd.DataFrame(data)
        
        # Save data
        file_path = os.path.join(feature_path, f"data_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.parquet")
        df.to_parquet(file_path)
        
        # Update metadata
        self.features[name].updated_at = datetime.utcnow()
        self._save_feature_metadata(self.features[name])
    
    async def get_feature(self, name: str, filters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Get feature data.
        
        Args:
            name: Feature name
            filters: Optional filters to apply
            
        Returns:
            Feature data as DataFrame
        """
        if name not in self.features:
            raise ValueError(f"Feature {name} not registered")
        
        feature_path = os.path.join(self.base_path, name)
        
        # Read all data files
        dfs = []
        for file_name in os.listdir(feature_path):
            if file_name.endswith('.parquet'):
                file_path = os.path.join(feature_path, file_name)
                df = pd.read_parquet(file_path)
                dfs.append(df)
        
        if not dfs:
            return pd.DataFrame()
        
        # Combine data
        data = pd.concat(dfs, ignore_index=True)
        
        # Apply filters
        if filters:
            for field, value in filters.items():
                if field in data.columns:
                    data = data[data[field] == value]
        
        return data
    
    def fit_scaler(self, name: str, data: pd.DataFrame) -> None:
        """
        Fit feature scaler.
        
        Args:
            name: Feature name
            data: Feature data
        """
        if name not in self.features:
            raise ValueError(f"Feature {name} not registered")
        
        scaler = StandardScaler()
        scaler.fit(data)
        
        # Save scaler
        scaler_path = os.path.join(self.base_path, name, 'scaler.joblib')
        joblib.dump(scaler, scaler_path)
        
        self.scalers[name] = scaler
    
    def transform_feature(self, name: str, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform feature data using scaler.
        
        Args:
            name: Feature name
            data: Feature data
            
        Returns:
            Transformed data
        """
        if name not in self.scalers:
            scaler_path = os.path.join(self.base_path, name, 'scaler.joblib')
            if os.path.exists(scaler_path):
                self.scalers[name] = joblib.load(scaler_path)
            else:
                raise ValueError(f"Scaler for feature {name} not found")
        
        return pd.DataFrame(
            self.scalers[name].transform(data),
            columns=data.columns,
            index=data.index
        )
    
    def list_features(self) -> List[Dict[str, Any]]:
        """
        List registered features.
        
        Returns:
            List of feature metadata
        """
        return [
            {
                'name': config.name,
                'description': config.description,
                'data_type': config.data_type,
                'tags': config.tags,
                'version': config.version,
                'created_at': config.created_at.isoformat(),
                'updated_at': config.updated_at.isoformat()
            }
            for config in self.features.values()
        ]
    
    def get_feature_metadata(self, name: str) -> Dict[str, Any]:
        """
        Get feature metadata.
        
        Args:
            name: Feature name
            
        Returns:
            Feature metadata
        """
        if name not in self.features:
            raise ValueError(f"Feature {name} not registered")
        
        config = self.features[name]
        return {
            'name': config.name,
            'description': config.description,
            'data_type': config.data_type,
            'tags': config.tags,
            'version': config.version,
            'created_at': config.created_at.isoformat(),
            'updated_at': config.updated_at.isoformat()
        }

class DriftDetector:
    """Detect data drift in features."""
    
    def __init__(self, window_size: int = 1000, threshold: float = 0.05):
        """
        Initialize drift detector.
        
        Args:
            window_size: Size of reference window
            threshold: Drift detection threshold
        """
        self.window_size = window_size
        self.threshold = threshold
        self.reference_data = {}
        self.drift_scores = {}
    
    def update_reference(self, name: str, data: pd.Series) -> None:
        """
        Update reference data for a feature.
        
        Args:
            name: Feature name
            data: Feature data
        """
        if len(data) < self.window_size:
            raise ValueError(f"Data length {len(data)} less than window size {self.window_size}")
        
        self.reference_data[name] = data[-self.window_size:]
    
    def detect_drift(self, name: str, data: pd.Series) -> Dict[str, Any]:
        """
        Detect drift in feature data.
        
        Args:
            name: Feature name
            data: Feature data to check
            
        Returns:
            Drift detection results
        """
        if name not in self.reference_data:
            raise ValueError(f"No reference data for feature {name}")
        
        reference = self.reference_data[name]
        
        # Calculate drift metrics
        ks_stat, p_value = stats.ks_2samp(reference, data)
        
        # Calculate distribution statistics
        ref_mean = reference.mean()
        ref_std = reference.std()
        data_mean = data.mean()
        data_std = data.std()
        
        # Calculate drift score
        drift_score = abs(data_mean - ref_mean) / ref_std
        
        # Store drift score
        self.drift_scores[name] = drift_score
        
        return {
            'feature': name,
            'drift_detected': p_value < self.threshold,
            'drift_score': drift_score,
            'p_value': p_value,
            'ks_statistic': ks_stat,
            'reference_mean': ref_mean,
            'reference_std': ref_std,
            'data_mean': data_mean,
            'data_std': data_std
        }
    
    def get_drift_scores(self) -> Dict[str, float]:
        """
        Get drift scores for all features.
        
        Returns:
            Dictionary of feature drift scores
        """
        return dict(self.drift_scores)
    
    def reset(self, name: Optional[str] = None) -> None:
        """
        Reset drift detection state.
        
        Args:
            name: Optional feature name to reset
        """
        if name:
            if name in self.reference_data:
                del self.reference_data[name]
            if name in self.drift_scores:
                del self.drift_scores[name]
        else:
            self.reference_data.clear()
            self.drift_scores.clear()

class PolarsProcessor:
    """High-performance data processor using Polars."""
    
    def __init__(self, config: ProcessingConfig):
        """Initialize processor."""
        self.config = config
        self.executor = ThreadPoolExecutor(max_workers=config.num_threads)
        self._processing_tasks: Dict[str, asyncio.Task] = {}
        
    async def process_batch(self, data: List[Dict[str, Any]]) -> pl.DataFrame:
        """Process a batch of data using Polars."""
        try:
            # Convert to Polars DataFrame
            df = pl.DataFrame(data)
            
            # Run processing in thread pool
            loop = asyncio.get_event_loop()
            processed_df = await loop.run_in_executor(
                self.executor,
                self._process_dataframe,
                df
            )
            
            return processed_df
            
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            raise
    
    def _process_dataframe(self, df: pl.DataFrame) -> pl.DataFrame:
        """Process DataFrame with Polars operations."""
        try:
            # Ensure timestamp column is datetime
            if 'timestamp' in df.columns:
                df = df.with_columns([
                    pl.col('timestamp').cast(pl.Datetime)
                ])
            
            # Sort by timestamp
            df = df.sort('timestamp')
            
            # Calculate basic statistics
            df = df.with_columns([
                # Price changes
                pl.col('price').diff().alias('price_change'),
                pl.col('price').pct_change().alias('price_return'),
                
                # Volume analysis
                pl.col('volume').rolling_mean(
                    window_size=self.config.window_size,
                    min_periods=self.config.min_periods
                ).alias('volume_ma'),
                
                # Volatility
                pl.col('price').rolling_std(
                    window_size=self.config.window_size,
                    min_periods=self.config.min_periods
                ).alias('volatility'),
                
                # VWAP
                (pl.col('price') * pl.col('volume')).rolling_sum(
                    window_size=self.config.window_size,
                    min_periods=self.config.min_periods
                ).alias('vwap_numerator'),
                pl.col('volume').rolling_sum(
                    window_size=self.config.window_size,
                    min_periods=self.config.min_periods
                ).alias('vwap_denominator')
            ])
            
            # Calculate VWAP
            df = df.with_columns([
                (pl.col('vwap_numerator') / pl.col('vwap_denominator')).alias('vwap')
            ])
            
            # Drop intermediate columns
            df = df.drop(['vwap_numerator', 'vwap_denominator'])
            
            # Add technical indicators
            df = self._add_technical_indicators(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error in DataFrame processing: {str(e)}")
            raise
    
    def _add_technical_indicators(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add technical indicators to DataFrame."""
        try:
            # Moving averages
            df = df.with_columns([
                pl.col('price').rolling_mean(
                    window_size=20,
                    min_periods=1
                ).alias('sma_20'),
                pl.col('price').rolling_mean(
                    window_size=50,
                    min_periods=1
                ).alias('sma_50'),
                pl.col('price').rolling_mean(
                    window_size=200,
                    min_periods=1
                ).alias('sma_200')
            ])
            
            # RSI
            delta = pl.col('price').diff()
            gain = delta.filter(delta > 0).fill_null(0)
            loss = -delta.filter(delta < 0).fill_null(0)
            
            avg_gain = gain.rolling_mean(window_size=14, min_periods=1)
            avg_loss = loss.rolling_mean(window_size=14, min_periods=1)
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            df = df.with_columns([
                rsi.alias('rsi')
            ])
            
            # MACD
            ema12 = pl.col('price').ewm_mean(span=12)
            ema26 = pl.col('price').ewm_mean(span=26)
            macd = ema12 - ema26
            signal = macd.ewm_mean(span=9)
            
            df = df.with_columns([
                macd.alias('macd'),
                signal.alias('macd_signal'),
                (macd - signal).alias('macd_histogram')
            ])
            
            # Bollinger Bands
            df = df.with_columns([
                pl.col('price').rolling_mean(
                    window_size=20,
                    min_periods=1
                ).alias('bb_middle'),
                (pl.col('price').rolling_mean(
                    window_size=20,
                    min_periods=1
                ) + 2 * pl.col('price').rolling_std(
                    window_size=20,
                    min_periods=1
                )).alias('bb_upper'),
                (pl.col('price').rolling_mean(
                    window_size=20,
                    min_periods=1
                ) - 2 * pl.col('price').rolling_std(
                    window_size=20,
                    min_periods=1
                )).alias('bb_lower')
            ])
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {str(e)}")
            raise
    
    async def process_stream(self, data_stream: asyncio.Queue) -> asyncio.Queue:
        """Process streaming data."""
        output_queue = asyncio.Queue()
        
        async def _process_stream():
            while True:
                try:
                    # Get data from input queue
                    data = await data_stream.get()
                    
                    # Process data
                    processed_data = await self.process_batch(data)
                    
                    # Put processed data in output queue
                    await output_queue.put(processed_data)
                    
                    # Mark task as done
                    data_stream.task_done()
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in stream processing: {str(e)}")
                    continue
        
        # Start processing task
        task = asyncio.create_task(_process_stream())
        self._processing_tasks['stream'] = task
        
        return output_queue
    
    async def stop(self):
        """Stop all processing tasks."""
        # Cancel all processing tasks
        for task in self._processing_tasks.values():
            task.cancel()
        
        # Wait for tasks to complete
        if self._processing_tasks:
            await asyncio.gather(*self._processing_tasks.values(), return_exceptions=True)
        
        # Shutdown thread pool
        self.executor.shutdown(wait=True)
        
        # Clear tasks
        self._processing_tasks.clear() 