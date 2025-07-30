"""Data models for tick analysis."""

from decimal import Decimal
from datetime import datetime, timezone

class Candle:
    def __init__(self, symbol, timeframe, open, high, low, close, volume, timestamp):
        self.symbol = symbol
        self.timeframe = timeframe
        self.open = Decimal(str(open))
        self.high = Decimal(str(high))
        self.low = Decimal(str(low))
        self.close = Decimal(str(close))
        self.volume = Decimal(str(volume))
        if isinstance(timestamp, (int, float)):
            self.timestamp = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)
        else:
            self.timestamp = timestamp

from enum import Enum

class Timeframe(Enum):
    ONE_MINUTE = "1m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    THIRTY_MINUTES = "30m"
    ONE_HOUR = "1h"
    FOUR_HOURS = "4h"
    ONE_DAY = "1d"
    ONE_WEEK = "1w"
    ONE_MONTH = "1M"

from tick_analysis.execution.order import Order, OrderType, OrderSide
# Re-export Position for compatibility with tests
from tick_analysis.portfolio.position import Position

