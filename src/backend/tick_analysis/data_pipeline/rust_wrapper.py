"""
Python wrapper for Rust-based high-performance processing functions.
"""

import logging
from typing import Dict, List, Any, Optional, Union
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    from .rust_processing import (
        calculate_technical_indicators,
        calculate_statistics,
        calculate_correlation
    )
    RUST_AVAILABLE = True
except ImportError:
    logger.warning("Rust processing module not available. Falling back to Python implementation.")
    RUST_AVAILABLE = False

@dataclass
class RustProcessingConfig:
    """Configuration for Rust-based processing."""
    window_size: int = 100
    min_periods: int = 20
    use_rust: bool = True

class RustProcessor:
    """High-performance data processor using Rust bindings."""
    
    def __init__(self, config: RustProcessingConfig):
        """Initialize processor."""
        self.config = config
        self.use_rust = config.use_rust and RUST_AVAILABLE
    
    def process_data(self, prices: np.ndarray, volumes: np.ndarray) -> Dict[str, np.ndarray]:
        """Process price and volume data using Rust functions."""
        try:
            if self.use_rust:
                return self._process_with_rust(prices, volumes)
            else:
                return self._process_with_python(prices, volumes)
                
        except Exception as e:
            logger.error(f"Error in data processing: {str(e)}")
            raise
    
    def _process_with_rust(self, prices: np.ndarray, volumes: np.ndarray) -> Dict[str, np.ndarray]:
        """Process data using Rust implementation."""
        # Calculate technical indicators
        indicators = calculate_technical_indicators(
            prices,
            volumes,
            self.config.window_size
        )
        
        # Calculate statistics
        stats = calculate_statistics(
            np.column_stack([prices, volumes]),
            axis=0
        )
        
        # Calculate correlation
        correlation = calculate_correlation(
            np.column_stack([prices, volumes])
        )
        
        # Combine results
        results = {
            **indicators,
            'statistics': stats,
            'correlation': correlation
        }
        
        return results
    
    def _process_with_python(self, prices: np.ndarray, volumes: np.ndarray) -> Dict[str, np.ndarray]:
        """Fallback to Python implementation."""
        # This is a simplified version of the processing logic
        # In practice, you would want to implement the full processing pipeline in Python
        
        # Calculate basic statistics
        stats = {
            'mean': np.mean(prices),
            'std': np.std(prices),
            'min': np.min(prices),
            'max': np.max(prices)
        }
        
        # Calculate simple moving average
        sma = np.convolve(
            prices,
            np.ones(self.config.window_size) / self.config.window_size,
            mode='valid'
        )
        
        # Calculate VWAP
        vwap = np.sum(prices * volumes) / np.sum(volumes)
        
        return {
            'sma': sma,
            'vwap': vwap,
            'statistics': stats
        }
    
    def calculate_custom_indicators(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate custom technical indicators."""
        if not self.use_rust:
            return self._calculate_custom_indicators_python(data)
        
        try:
            # Calculate statistics
            stats = calculate_statistics(data, axis=0)
            
            # Calculate correlation
            correlation = calculate_correlation(data)
            
            return {
                'statistics': stats,
                'correlation': correlation
            }
            
        except Exception as e:
            logger.error(f"Error calculating custom indicators: {str(e)}")
            return self._calculate_custom_indicators_python(data)
    
    def _calculate_custom_indicators_python(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate custom indicators using Python."""
        # Calculate basic statistics
        stats = {
            'mean': np.mean(data, axis=0),
            'std': np.std(data, axis=0),
            'min': np.min(data, axis=0),
            'max': np.max(data, axis=0)
        }
        
        # Calculate correlation matrix
        correlation = np.corrcoef(data.T)
        
        return {
            'statistics': stats,
            'correlation': correlation
        } 