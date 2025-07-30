"""
Feature Engineering Module

This module provides feature engineering utilities for financial time series data,
including feature creation, transformation, and selection.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Tuple, Any
from dataclasses import dataclass
from enum import Enum, auto
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

class FeatureType(Enum):
    """Types of features in the feature engineering pipeline."""
    PRICE = auto()
    VOLUME = auto()
    VOLATILITY = auto()
    MOMENTUM = auto()
    TREND = auto()
    PATTERN = auto()
    STATISTICAL = auto()
    CUSTOM = auto()

@dataclass
class FeatureConfig:
    """Configuration for a feature in the feature engineering pipeline."""
    name: str
    feature_type: FeatureType
    params: dict
    description: str = ""
    enabled: bool = True

class FeatureEngineer:
    """
    Feature engineering pipeline for financial time series data.
    
    This class provides methods to create, transform, and select features
    from OHLCV data and technical indicators.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the feature engineering pipeline.
        
        Args:
            config: Optional configuration dictionary for feature engineering
        """
        self.config = config or {}
        self.features: List[FeatureConfig] = []
        self._initialize_default_features()
    
    def _initialize_default_features(self) -> None:
        """Initialize default feature configurations."""
        # Price features
        self._add_feature("returns", FeatureType.PRICE, {"period": 1}, "Simple returns")
        self._add_feature("log_returns", FeatureType.PRICE, {"period": 1}, "Logarithmic returns")
        
        # Volume features
        self._add_feature("volume_change", FeatureType.VOLUME, {"period": 1}, "Volume change")
        self._add_feature("volume_ma", FeatureType.VOLUME, {"period": 20}, "Volume moving average")
        
        # Volatility features
        self._add_feature("volatility", FeatureType.VOLATILITY, {"period": 20}, "Rolling standard deviation of returns")
        self._add_feature("atr", FeatureType.VOLATILITY, {"period": 14}, "Average True Range")
        
        # Momentum features
        self._add_feature("rsi", FeatureType.MOMENTUM, {"period": 14}, "Relative Strength Index")
        self._add_feature("macd", FeatureType.MOMENTUM, {"fast": 12, "slow": 26, "signal": 9}, "Moving Average Convergence Divergence")
        
        # Trend features
        self._add_feature("sma", FeatureType.TREND, {"period": 20}, "Simple Moving Average")
        self._add_feature("ema", FeatureType.TREND, {"period": 20}, "Exponential Moving Average")
    
    def _add_feature(self, name: str, feature_type: FeatureType, params: Dict, 
                    description: str = "", enabled: bool = True) -> None:
        """
        Add a feature configuration to the pipeline.
        
        Args:
            name: Feature name
            feature_type: Type of feature (from FeatureType enum)
            params: Dictionary of parameters for the feature
            description: Optional description of the feature
            enabled: Whether the feature is enabled by default
        """
        self.features.append(
            FeatureConfig(
                name=name,
                feature_type=feature_type,
                params=params,
                description=description,
                enabled=enabled
            )
        )
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from OHLCV data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added features
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
            
        # Make a copy to avoid modifying the original DataFrame
        features_df = df.copy()
        
        # Add basic price features
        features_df = self._add_price_features(features_df)
        
        # Add volume features
        features_df = self._add_volume_features(features_df)
        
        # Add volatility features
        features_df = self._add_volatility_features(features_df)
        
        # Add momentum features
        features_df = self._add_momentum_features(features_df)
        
        # Add trend features
        features_df = self._add_trend_features(features_df)
        
        # Add statistical features
        features_df = self._add_statistical_features(features_df)
        
        return features_df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features."""
        features = df.copy()
        
        # Returns
        if self._is_feature_enabled("returns"):
            period = self._get_feature_param("returns", "period", 1)
            features[f'returns_{period}'] = df['close'].pct_change(periods=period)
        
        # Log returns
        if self._is_feature_enabled("log_returns"):
            period = self._get_feature_param("log_returns", "period", 1)
            features[f'log_returns_{period}'] = np.log(df['close'] / df['close'].shift(period))
        
        # Price ranges
        features['range'] = df['high'] - df['low']
        features['body'] = (df['close'] - df['open']).abs()
        features['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        features['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        
        # Price relative to moving averages
        if self._is_feature_enabled("sma"):
            period = self._get_feature_param("sma", "period", 20)
            sma = df['close'].rolling(window=period).mean()
            features[f'price_sma_ratio_{period}'] = df['close'] / sma - 1
        
        return features
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        features = df.copy()
        
        # Volume change
        if self._is_feature_enabled("volume_change"):
            period = self._get_feature_param("volume_change", "period", 1)
            features[f'volume_change_{period}'] = df['volume'].pct_change(periods=period)
        
        # Volume moving average
        if self._is_feature_enabled("volume_ma"):
            period = self._get_feature_param("volume_ma", "period", 20)
            features[f'volume_ma_{period}'] = df['volume'].rolling(window=period).mean()
            features[f'volume_ratio_{period}'] = df['volume'] / features[f'volume_ma_{period}']
        
        return features
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based features."""
        features = df.copy()
        
        # Historical volatility
        if self._is_feature_enabled("volatility"):
            period = self._get_feature_param("volatility", "period", 20)
            returns = df['close'].pct_change()
            features[f'volatility_{period}'] = returns.rolling(window=period).std() * np.sqrt(252)  # Annualized
        
        # Average True Range (ATR)
        if self._is_feature_enabled("atr"):
            period = self._get_feature_param("atr", "period", 14)
            high = df['high']
            low = df['low']
            close = df['close']
            
            tr1 = high - low
            tr2 = (high - close.shift()).abs()
            tr3 = (low - close.shift()).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            features[f'atr_{period}'] = tr.rolling(window=period).mean()
        
        return features
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based features."""
        features = df.copy()
        close = df['close']
        
        # RSI
        if self._is_feature_enabled("rsi"):
            period = self._get_feature_param("rsi", "period", 14)
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / (loss + 1e-10)  # Avoid division by zero
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # MACD
        if self._is_feature_enabled("macd"):
            fast = self._get_feature_param("macd", "fast", 12)
            slow = self._get_feature_param("macd", "slow", 26)
            signal = self._get_feature_param("macd", "signal", 9)
            
            ema_fast = close.ewm(span=fast, adjust=False).mean()
            ema_slow = close.ewm(span=slow, adjust=False).mean()
            macd = ema_fast - ema_slow
            signal_line = macd.ewm(span=signal, adjust=False).mean()
            
            features['macd'] = macd
            features['macd_signal'] = signal_line
            features['macd_hist'] = macd - signal_line
        
        return features
    
    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend-based features."""
        features = df.copy()
        close = df['close']
        
        # Simple Moving Average
        if self._is_feature_enabled("sma"):
            period = self._get_feature_param("sma", "period", 20)
            features[f'sma_{period}'] = close.rolling(window=period).mean()
        
        # Exponential Moving Average
        if self._is_feature_enabled("ema"):
            period = self._get_feature_param("ema", "period", 20)
            features[f'ema_{period}'] = close.ewm(span=period, adjust=False).mean()
        
        return features
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features."""
        features = df.copy()
        
        # Z-Score
        window = 20
        returns = df['close'].pct_change()
        features['zscore'] = (returns - returns.rolling(window=window).mean()) / returns.rolling(window=window).std()
        
        # Rolling quantiles
        for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
            features[f'quantile_{int(q*100)}'] = df['close'].rolling(window=window).quantile(q)
        
        return features
    
    def _is_feature_enabled(self, feature_name: str) -> bool:
        """Check if a feature is enabled in the configuration."""
        for feature in self.features:
            if feature.name == feature_name:
                return feature.enabled
        return False
    
    def _get_feature_param(self, feature_name: str, param_name: str, default: Any) -> Any:
        """Get a parameter value for a feature."""
        for feature in self.features:
            if feature.name == feature_name:
                return feature.params.get(param_name, default)
        return default
    
    def get_feature_list(self) -> List[str]:
        """
        Get a list of all feature names.
        
        Returns:
            List of feature names
        """
        return [col for col in self.create_features(pd.DataFrame()).columns 
                if col not in ['open', 'high', 'low', 'close', 'volume']]

# Example usage
if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    
    # Download some sample data
    data = yf.download('AAPL', start='2020-01-01', end='2021-01-01')
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer()
    
    # Create features
    features = feature_engineer.create_features(data)
    
    print(f"Original columns: {list(data.columns)}")
    print(f"Generated features: {feature_engineer.get_feature_list()}")
    print(f"Features DataFrame shape: {features.shape}")
