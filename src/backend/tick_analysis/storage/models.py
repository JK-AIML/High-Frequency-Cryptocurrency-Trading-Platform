"""
Data models for time-series storage.

Defines the schema and data models used for storing tick data in the time-series database.
"""

from datetime import datetime
from typing import Dict, List, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field, validator
from typing_extensions import Literal


class TimeRange(BaseModel):
    """Represents a time range for querying time-series data."""
    start_time: datetime
    end_time: datetime
    
    @validator('end_time')
    def validate_time_range(cls, v, values):
        if 'start_time' in values and v <= values['start_time']:
            raise ValueError('end_time must be after start_time')
        return v


class AggregationType(str, Enum):
    """Supported aggregation types for time-series queries."""
    NONE = 'none'
    MINUTE = '1m'
    FIVE_MINUTES = '5m'
    FIFTEEN_MINUTES = '15m'
    HOUR = '1h'
    DAY = '1d'


class TickDataPoint(BaseModel):
    """Represents a single tick data point."""
    timestamp: datetime
    symbol: str
    price: float
    volume: float
    exchange: str
    bid: Optional[float] = None
    ask: Optional[float] = None
    trade_id: Optional[str] = None
    metadata: Dict[str, Union[str, float, int, bool, None]] = Field(default_factory=dict)


class OHLCV(BaseModel):
    """Open-High-Low-Close-Volume data point."""
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: Optional[float] = None  # Volume Weighted Average Price
    trade_count: Optional[int] = None
    exchange: Optional[str] = None


class OrderBookSnapshot(BaseModel):
    """Order book snapshot at a specific point in time."""
    timestamp: datetime
    symbol: str
    exchange: str
    bids: List[Dict[float, float]]  # price -> quantity
    asks: List[Dict[float, float]]  # price -> quantity
    
    @validator('bids', 'asks')
    def validate_price_quantity_pairs(cls, v):
        for item in v:
            if len(item) != 1:
                raise ValueError('Each price-quantity pair must have exactly one key-value pair')
        return v


class Trade(BaseModel):
    """Trade execution record."""
    trade_id: str
    timestamp: datetime
    symbol: str
    price: float
    quantity: float
    side: Literal['buy', 'sell']
    exchange: str
    taker_side: Optional[Literal['buy', 'sell']] = None
    trade_condition: Optional[str] = None
    is_snapshot: bool = False


class MarketDepth(BaseModel):
    """Market depth data."""
    timestamp: datetime
    symbol: str
    exchange: str
    levels: int
    bids: List[Dict[float, float]]  # price -> quantity
    asks: List[Dict[float, float]]  # price -> quantity
    
    class Config:
        json_encoders = {
            'datetime': lambda v: v.isoformat()
        }
