"""Order class for tick analysis."""

from decimal import Decimal
from datetime import datetime, timezone
from enum import Enum, auto
import uuid

class OrderStatus(Enum):
    NEW = auto()
    FILLED = auto()
    PARTIALLY_FILLED = auto()
    REJECTED = auto()
    CANCELED = auto()

class OrderSide(Enum):
    BUY = auto()
    SELL = auto()

class OrderType(Enum):
    MARKET = auto()
    LIMIT = auto()
    STOP = auto()
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    IOC = "ioc"
    FOK = "fok"

class TimeInForce(Enum):
    GTC = auto()  # Good Till Cancel
    IOC = auto()  # Immediate or Cancel
    FOK = auto()  # Fill or Kill

class PositionSide(Enum):
    LONG = auto()
    SHORT = auto()

class Order:
    def __init__(self, symbol=None, side=None, order_type=None, quantity=None, price=None, 
                 limit_price=None, stop_price=None, time_in_force=None, timestamp=None, **kwargs):
        self.id = str(uuid.uuid4())  # Generate a unique ID for each order
        self.symbol = symbol
        self.side = side
        self.order_type = order_type
        self.quantity = Decimal(str(quantity)) if quantity is not None else None
        self.price = Decimal(str(price)) if price is not None else None
        self.limit_price = Decimal(str(limit_price)) if limit_price is not None else None
        self.stop_price = Decimal(str(stop_price)) if stop_price is not None else None
        self.time_in_force = time_in_force or TimeInForce.GTC
        self.timestamp = timestamp or datetime.now(timezone.utc)
        self.status = OrderStatus.NEW
        self.executed_quantity = Decimal('0')
        self.executed_price = None
        self.fee = Decimal('0')
        self._validate()

    def _validate(self):
        """Validate order attributes."""
        if not self.symbol:
            raise ValueError("Order must have a symbol")
        if not self.side:
            raise ValueError("Order must have a side")
        if not self.order_type:
            raise ValueError("Order must have an order type")
        if not self.quantity or self.quantity <= 0:
            raise ValueError("Order must have a positive quantity")
            
        if self.order_type == OrderType.LIMIT and not self.limit_price:
            raise ValueError("Limit orders must have a limit price")
        if self.order_type == OrderType.STOP and not self.stop_price:
            raise ValueError("Stop orders must have a stop price")
        if self.order_type == OrderType.STOP_LIMIT and (not self.stop_price or not self.limit_price):
            raise ValueError("Stop-limit orders must have both stop and limit prices")
            
        if self.limit_price is not None and self.limit_price <= 0:
            raise ValueError("Limit price must be positive")
        if self.stop_price is not None and self.stop_price <= 0:
            raise ValueError("Stop price must be positive")
        if self.price is not None and self.price <= 0:
            raise ValueError("Price must be positive")

    def is_filled(self):
        """Check if order is filled."""
        return self.status == OrderStatus.FILLED

    def is_partially_filled(self):
        """Check if order is partially filled."""
        return self.status == OrderStatus.PARTIALLY_FILLED

    def is_rejected(self):
        """Check if order is rejected."""
        return self.status == OrderStatus.REJECTED

    def is_canceled(self):
        """Check if order is canceled."""
        return self.status == OrderStatus.CANCELED

    def is_pending(self):
        """Check if order is pending (NEW status)."""
        return self.status == OrderStatus.NEW

    def get_execution_price(self):
        """Get the execution price of the order."""
        return self.executed_price if self.executed_price is not None else self.price

    def get_execution_quantity(self):
        """Get the execution quantity of the order."""
        return self.executed_quantity if self.executed_quantity > 0 else self.quantity

    def get_total_cost(self):
        """Get the total cost of the order including fees."""
        if not self.is_filled() and not self.is_partially_filled():
            return Decimal('0')
        return (self.get_execution_price() * self.get_execution_quantity()) + self.fee

    def update_status(self, status):
        """Update order status."""
        self.status = status

    def update_execution(self, executed_price, executed_quantity, fee=None):
        """Update order execution details."""
        self.executed_price = Decimal(str(executed_price))
        self.executed_quantity = Decimal(str(executed_quantity))
        if fee is not None:
            self.fee = Decimal(str(fee))

class Trade:
    def __init__(self, order):
        if order is None:
            raise ValueError("Order cannot be None")
        self.symbol = order.symbol
        self.side = order.side
        self.quantity = order.quantity
        self.price = order.price
        self.order_type = order.order_type
        self.timestamp = datetime.now(timezone.utc)
        self.realized_pnl = Decimal('0')
    
    def __eq__(self, other):
        if not isinstance(other, Trade):
            raise TypeError("Cannot compare Trade with non-Trade type")
        return (self.symbol == other.symbol and
                self.side == other.side and
                self.quantity == other.quantity and
                self.price == other.price and
                self.order_type == other.order_type and
                self.timestamp == other.timestamp)
    
    def __hash__(self):
        return hash((self.symbol, self.side, self.quantity, self.price,
                    self.order_type, self.timestamp))


