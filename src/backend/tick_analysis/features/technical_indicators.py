"""
Comprehensive Technical Indicators Module

Provides 50+ technical indicators for financial analysis and trading strategies.
Uses pandas_ta and ta libraries for calculations.
"""

"""
Comprehensive Technical Indicators Module

This module provides a wide range of technical indicators for financial analysis.
It integrates multiple indicator libraries and provides a unified interface.
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum, auto
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Import TA-Lib if available
try:
    import talib
except ImportError:
    talib = None
    print("Warning: TA-Lib not installed. Some indicators may not be available.")

# Import ta library
try:
    from ta import add_all_ta_features
    from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator, ROCIndicator, TSIIndicator
    from ta.trend import MACD, ADXIndicator, AroonIndicator, CCIIndicator, DPOIndicator, KSTIndicator
    from ta.volatility import BollingerBands, AverageTrueRange, DonchianChannel, KeltnerChannel
    from ta.volume import AccDistIndex, EaseOfMovement, MFIIndicator, OnBalanceVolumeIndicator
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    print("Warning: ta library not installed. Some indicators may not be available.")

class IndicatorType(Enum):
    """Types of technical indicators."""
    TREND = auto()
    MOMENTUM = auto()
    VOLATILITY = auto()
    VOLUME = auto()
    CUSTOM = auto()

@dataclass
class IndicatorConfig:
    """Configuration for a technical indicator."""
    name: str
    indicator_type: IndicatorType
    params: dict
    description: str = ""
    source: str = "custom"  # 'ta', 'pandas_ta', 'talib', or 'custom'

class TechnicalIndicators:
    """
    Comprehensive technical indicators calculator with 50+ indicators.
    
    This class provides a unified interface for calculating technical indicators
    from multiple libraries (pandas_ta, ta-lib, ta) with consistent return formats.
    """
    
    def __init__(self, df: pd.DataFrame, **kwargs):
        """
        Initialize with OHLCV DataFrame.
        
        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
            **kwargs: Additional configuration parameters
        """
        self.df = df.copy()
        self.config = kwargs.get('config', {})
        self._validate_input()
        self._indicators = []
        
    def _validate_input(self) -> None:
        """Validate input DataFrame has required columns."""
        required = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required if col not in self.df.columns
                  if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    def add_all_indicators(self) -> pd.DataFrame:
        """
        Add all available technical indicators to the DataFrame.
        
        Returns:
            DataFrame with all indicators added as new columns
        """
        # Add indicators from different sources
        if TA_AVAILABLE:
            self._add_ta_indicators()
            
        self._add_pandas_ta_indicators()
        
        # Add custom indicators
        self._add_custom_indicators()
        
        # Add TA-Lib indicators if available
        if talib is not None:
            self._add_talib_indicators()
        
        return self.df
    
    def _add_ta_indicators(self) -> None:
        """Add indicators from the ta library."""
        if not TA_AVAILABLE:
            return
            
        try:
            # Add all ta features
            self.df = add_all_ta_features(
                self.df, 
                open="open", 
                high="high", 
                low="low", 
                close="close", 
                volume="volume",
                fillna=True
            )
            
            # Add custom ta indicators
            self._add_custom_ta_indicators()
            
        except Exception as e:
            print(f"Error adding ta indicators: {str(e)}")
    
    def _add_pandas_ta_indicators(self) -> None:
        """Add indicators from pandas_ta library."""
        try:
            # Add basic indicators
            self.df.ta.rsi(length=14, append=True)
            self.df.ta.macd(append=True)
            self.df.ta.bbands(append=True)
            self.df.ta.stoch(append=True)
            self.df.ta.atr(append=True)
            self.df.ta.obv(append=True)
            
            # Add additional indicators
            self._add_advanced_pandas_ta_indicators()
            
        except Exception as e:
            print(f"Error adding pandas_ta indicators: {str(e)}")
    
    def _add_talib_indicators(self) -> None:
        """Add indicators from TA-Lib if available."""
        if talib is None:
            return
            
        try:
            # Add TA-Lib indicators
            self.df['rsi_talib'] = talib.RSI(self.df['close'], timeperiod=14)
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(
                self.df['close'], 
                fastperiod=12, 
                slowperiod=26, 
                signalperiod=9
            )
            self.df['macd_talib'] = macd
            self.df['macd_signal_talib'] = macd_signal
            self.df['macd_hist_talib'] = macd_hist
            
            # Bollinger Bands
            upper, middle, lower = talib.BBANDS(
                self.df['close'], 
                timeperiod=20, 
                nbdevup=2, 
                nbdevdn=2
            )
            self.df['bb_upper_talib'] = upper
            self.df['bb_middle_talib'] = middle
            self.df['bb_lower_talib'] = lower
            
        except Exception as e:
            print(f"Error adding TA-Lib indicators: {str(e)}")
    
    def _add_advanced_pandas_ta_indicators(self) -> None:
        """Add more advanced indicators from pandas_ta."""
        try:
            # Advanced momentum indicators
            self.df.ta.cci(append=True)
            self.df.ta.willr(append=True)
            self.df.ta.roc(append=True)
            self.df.ta.tsi(append=True)
            
            # Trend indicators
            self.df.ta.adx(append=True)
            self.df.ta.aroon(append=True)
            self.df.ta.psar(append=True)
            
            # Volatility indicators
            self.df.ta.kc(append=True)  # Keltner Channels
            self.df.ta.donchian(append=True)
            
            # Volume indicators
            self.df.ta.ad(append=True)  # Accumulation/Distribution
            self.df.ta.adosc(append=True)  # AD Oscillator
            self.df.ta.mfi(append=True)  # Money Flow Index
            self.df.ta.cmf(append=True)  # Chaikin Money Flow
            
            # Custom indicators
            self.df.ta.supertrend(append=True)
            self.df.ta.hma(append=True)  # Hull Moving Average
            self.df.ta.zlma(append=True)  # Zero Lag Moving Average
            
        except Exception as e:
            print(f"Error adding advanced pandas_ta indicators: {str(e)}")
    
    def _add_custom_indicators(self) -> None:
        """Add custom technical indicators."""
        try:
            # Custom Volume Profile
            self.df['vp'] = (self.df['close'] * self.df['volume']).rolling(14).sum() / self.df['volume'].rolling(14).sum()
            
            # VWAP
            self.df['vwap'] = (self.df['volume'] * (self.df['high'] + self.df['low'] + self.df['close']) / 3).cumsum() / self.df['volume'].cumsum()
            
            # Custom RSI with different periods
            for period in [7, 14, 21]:
                self.df[f'rsi_{period}'] = self._calculate_rsi(self.df['close'], period)
            
            # Custom Moving Averages
            for period in [5, 10, 20, 50, 100, 200]:
                self.df[f'sma_{period}'] = self.df['close'].rolling(window=period).mean()
                self.df[f'ema_{period}'] = self.df['close'].ewm(span=period, adjust=False).mean()
            
            # Volatility
            self.df['returns'] = self.df['close'].pct_change()
            self.df['volatility_20'] = self.df['returns'].rolling(window=20).std() * np.sqrt(252)
            
            # Price Patterns
            self._add_price_patterns()
            
        except Exception as e:
            print(f"Error adding custom indicators: {str(e)}")
    
    def _add_price_patterns(self) -> None:
        """Add candlestick pattern recognition."""
        if talib is None:
            return
            
        try:
            # Common candlestick patterns
            patterns = [
                'CDL2CROWS', 'CDL3BLACKCROWS', 'CDL3INSIDE', 'CDL3LINESTRIKE',
                'CDL3OUTSIDE', 'CDL3STARSINSOUTH', 'CDL3WHITESOLDIERS', 'CDLABANDONEDBABY',
                'CDLADVANCEBLOCK', 'CDLBELTHOLD', 'CDLBREAKAWAY', 'CDLCLOSINGMARUBOZU',
                'CDLCONCEALBABYSWALL', 'CDLCOUNTERATTACK', 'CDLDARKCLOUDCOVER', 'CDLDOJI',
                'CDLDOJISTAR', 'CDLDRAGONFLYDOJI', 'CDLENGULFING', 'CDLEVENINGDOJISTAR',
                'CDLEVENINGSTAR', 'CDLGAPSIDESIDEWHITE', 'CDLGRAVESTONEDOJI', 'CDLHAMMER',
                'CDLHANGINGMAN', 'CDLHARAMI', 'CDLHARAMICROSS', 'CDLHIGHWAVE', 'CDLHIKKAKE',
                'CDLHIKKAKEMOD', 'CDLHOMINGPIGEON', 'CDLIDENTICAL3CROWS', 'CDLINNECK',
                'CDLINVERTEDHAMMER', 'CDLKICKING', 'CDLKICKINGBYLENGTH', 'CDLLADDERBOTTOM',
                'CDLLONGLEGGEDDOJI', 'CDLLONGLINE', 'CDLMARUBOZU', 'CDLMATCHINGLOW',
                'CDLMATHOLD', 'CDLMORNINGDOJISTAR', 'CDLMORNINGSTAR', 'CDLONNECK',
                'CDLPIERCING', 'CDLRICKSHAWMAN', 'CDLRISEFALL3METHODS', 'CDLSEPARATINGLINES',
                'CDLSHOOTINGSTAR', 'CDLSHORTLINE', 'CDLSPINNINGTOP', 'CDLSTALLEDPATTERN',
                'CDLSTICKSANDWICH', 'CDLTAKURI', 'CDLTASUKIGAP', 'CDLTHRUSTING',
                'CDLTRISTAR', 'CDLUNIQUE3RIVER', 'CDLUPSIDEGAP2CROWS', 'CDLXSIDEGAP3METHODS'
            ]
            
            for pattern in patterns:
                if hasattr(talib, pattern):
                    pattern_func = getattr(talib, pattern)
                    self.df[pattern.lower()] = pattern_func(
                        self.df['open'], self.df['high'], 
                        self.df['low'], self.df['close']
                    )
                    
        except Exception as e:
            print(f"Error adding price patterns: {str(e)}")
    
    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI using the standard formula."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)  # Avoid division by zero
        return 100 - (100 / (1 + rs))
    
    def _add_pandas_ta_indicators(self) -> None:
        """Add indicators from pandas_ta library."""
        # Momentum Indicators
        self.df.ta.rsi(append=True)
        self.df.ta.macd(append=True)
        self.df.ta.stoch(append=True)
        self.df.ta.cmo(append=True)
        self.df.ta.ppo(append=True)
        self.df.ta.roc(append=True)
        self.df.ta.willr(append=True)
        
        # Trend Indicators
        self.df.ta.adx(append=True)
        self.df.ta.cci(append=True)
        self.df.ta.dpo(append=True)
        self.df.ta.kst(append=True)
        self.df.ta.ichimoku(append=True)
        self.df.ta.vortex(append=True)
        
        # Volatility Indicators
        self.df.ta.bbands(append=True)
        self.df.ta.kc(append=True)  # Keltner Channels
        self.df.ta.donchian(append=True)
        self.df.ta.atr(append=True)
        
        # Volume Indicators
        self.df.ta.obv(append=True)
        self.df.ta.ad(append=True)  # Accumulation/Distribution
        self.df.ta.adosc(append=True)  # AD Oscillator
        self.df.ta.mfi(append=True)  # Money Flow Index
        self.df.ta.cmf(append=True)  # Chaikin Money Flow
        
        # Custom Indicators
        self.df.ta.supertrend(append=True)
        self.df.ta.hma(append=True)  # Hull Moving Average
        self.df.ta.zlma(append=True)  # Zero Lag Moving Average
    
    def _add_ta_indicators(self) -> None:
        """Add indicators from ta library."""
        # Add all ta features
        self.df = add_all_ta_features(
            self.df, 
            open="open", 
            high="high", 
            low="low", 
            close="close", 
            volume="volume",
            fillna=True
        )
    
    def _add_custom_indicators(self) -> None:
        """Add custom technical indicators."""
        # Custom Volume Profile
        self.df['vp'] = (self.df['close'] * self.df['volume']).rolling(14).sum() / self.df['volume'].rolling(14).sum()
        
        # VWAP
        self.df['vwap'] = (self.df['volume'] * (self.df['high'] + self.df['low'] + self.df['close']) / 3).cumsum() / self.df['volume'].cumsum()
        
        # Custom RSI with different periods
        for period in [7, 14, 21]:
            self.df[f'rsi_{period}'] = RSIIndicator(close=self.df['close'], window=period).rsi()
        
        # Custom MACD variations
        self.df['macd_5_35_5'], self.df['macdsignal_5_35_5'], _ = MACD(
            close=self.df['close'], 
            window_slow=35, 
            window_fast=5, 
            window_sign=5
        ).macd(), MACD(
            close=self.df['close'], 
            window_slow=35, 
            window_fast=5, 
            window_sign=5
        ).macd_signal(), MACD(
            close=self.df['close'], 
            window_slow=35, 
            window_fast=5, 
            window_sign=5
        ).macd_diff()
        
        # Custom Bollinger Bands
        bb = BollingerBands(close=self.df['close'], window=20, window_dev=2)
        self.df['bb_upper_band'] = bb.bollinger_hband()
        self.df['bb_middle_band'] = bb.bollinger_mavg()
        self.df['bb_lower_band'] = bb.bollinger_lband()
        self.df['bb_%b'] = (self.df['close'] - self.df['bb_lower_band']) / (self.df['bb_upper_band'] - self.df['bb_lower_band'])
        
        # Custom ATR
        self.df['atr_14'] = AverageTrueRange(
            high=self.df['high'], 
            low=self.df['low'], 
            close=self.df['close'], 
            window=14
        ).average_true_range()
        
        # Custom Stochastic RSI
        rsi = RSIIndicator(close=self.df['close'], window=14).rsi()
        stoch_rsi = (rsi - rsi.rolling(14).min()) / (rsi.rolling(14).max() - rsi.rolling(14).min())
        self.df['stoch_rsi'] = stoch_rsi
        
        # Custom Volume Indicators
        self.df['volume_ma_20'] = self.df['volume'].rolling(20).mean()
        self.df['volume_ratio'] = self.df['volume'] / self.df['volume_ma_20']
        
        # Price-Volume Trend
        self.df['pvt'] = ((self.df['close'] - self.df['close'].shift(1)) / self.df['close'].shift(1)) * self.df['volume']
        self.df['pvt'] = self.df['pvt'].cumsum()
        
        # Custom Trend Indicators
        self.df['sma_50'] = self.df['close'].rolling(50).mean()
        self.df['sma_200'] = self.df['close'].rolling(200).mean()
        self.df['sma_cross'] = np.where(self.df['sma_50'] > self.df['sma_200'], 1, -1)
        
        # Custom Volatility
        self.df['log_returns'] = np.log(self.df['close'] / self.df['close'].shift(1))
        self.df['realized_vol'] = self.df['log_returns'].rolling(window=20).std() * np.sqrt(252)
        
        # Custom Momentum
        self.df['roc_10'] = ROCIndicator(close=self.df['close'], window=10).roc()
        self.df['roc_20'] = ROCIndicator(close=self.df['close'], window=20).roc()
        
        # Custom Trend Strength
        self.df['adx'] = ADXIndicator(
            high=self.df['high'], 
            low=self.df['low'], 
            close=self.df['close'], 
            window=14
        ).adx()
        
        # Custom Volume Profile
        self.df['poc'] = self.df.groupby(
            pd.qcut(self.df['close'], 20)
        )['volume'].transform('sum')
        
        # Custom Support/Resistance
        self.df['resistance'] = self.df['high'].rolling(20).max()
        self.df['support'] = self.df['low'].rolling(20).min()
        
        # Custom Trend Following
        self.df['ema_9'] = self.df['close'].ewm(span=9, adjust=False).mean()
        self.df['ema_21'] = self.df['close'].ewm(span=21, adjust=False).mean()
        self.df['ema_50'] = self.df['close'].ewm(span=50, adjust=False).mean()
        
        # Custom Mean Reversion
        self.df['zscore'] = (self.df['close'] - self.df['close'].rolling(20).mean()) / self.df['close'].rolling(20).std()
        
        # Custom Volume Profile
        self.df['volume_profile'] = self.df['volume'] * (self.df['high'] - self.df['low']) / (self.df['high'] - self.df['low']).rolling(20).mean()


# Example usage
if __name__ == "__main__":
    # Example DataFrame with OHLCV data
    data = {
        'open': [100, 101, 102, 101, 103],
        'high': [101, 102, 103, 104, 105],
        'low': [99, 100, 101, 100, 102],
        'close': [100.5, 101.5, 102.5, 103.5, 104.5],
        'volume': [1000, 1200, 1500, 1300, 2000]
    }
    df = pd.DataFrame(data)
    
    # Add all indicators
    ti = TechnicalIndicators(df)
    df_with_indicators = ti.add_all_indicators()
    
    print(f"Added {len(df_with_indicators.columns) - 5} technical indicators")
    print("Available indicators:", list(df_with_indicators.columns))
