from decimal import Decimal
from tick_analysis.execution.order import PositionSide, OrderSide




__all__ = ["PositionSide", "Position"]

class Position:
    def __init__(self, symbol=None, quantity=0, entry_price=0, side=None, realized_pnl=0, amount=None, **kwargs):
        from tick_analysis.execution.order import PositionSide
        # Support both 'quantity' and 'amount' for test compatibility
        if amount is not None:
            self.quantity = Decimal(amount)
        else:
            self.quantity = Decimal(quantity)
        self.symbol = symbol
        self.entry_price = Decimal(entry_price)
        self.side = PositionSide.LONG if side is None else side
        self.realized_pnl = Decimal(realized_pnl)
        self.current_price = Decimal(entry_price)
        self.unrealized_pnl = Decimal('0')
        self.last_update_time = None

    def update_market_value(self, price):
        """Update position's market value and unrealized PnL."""
        if price is None:
            if self.current_price is None:
                self.current_price = self.entry_price
            price = self.current_price
        else:
            price = Decimal(price)
            if price < 0:
                raise ValueError("Price cannot be negative")
            self.current_price = price
        if self.quantity != 0:
            if self.side == PositionSide.SHORT:
                self.unrealized_pnl = (self.entry_price - self.current_price) * abs(self.quantity)
            else:
                self.unrealized_pnl = (self.current_price - self.entry_price) * self.quantity
        else:
            self.unrealized_pnl = Decimal('0')
        from datetime import datetime, timezone
        self.last_update_time = datetime.now(timezone.utc)

    def reset(self):
        """Reset position state."""
        self.quantity = Decimal('0')
        self.entry_price = Decimal('0')
        self.current_price = Decimal('0')
        self.side = PositionSide.LONG  # Always set to LONG after reset to match test expectation
        self.unrealized_pnl = Decimal('0')
        # Don't reset realized_pnl as it's cumulative

    def get_market_value(self):
        """Get current market value of position."""
        return self.quantity * self.current_price if self.quantity > 0 else Decimal('0')

    def get_cost_basis(self):
        """Get total cost basis of position."""
        return self.quantity * self.entry_price if self.quantity > 0 else Decimal('0')

    def get_total_pnl(self):
        """Get total PnL (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl

    def add_trade(self, order, decrement=True):
        """Add a trade to the position and update state."""
        if order is None:
            raise ValueError("Trade order cannot be None")
        executed_price = getattr(order, 'executed_price', None)
        if executed_price is None:
            executed_price = getattr(order, 'price', None)
        if executed_price is None:
            raise ValueError("Order must have an executed_price or price")
        executed_price = Decimal(executed_price)
        side = order.side
        qty = order.get_execution_quantity()  # Use method to get correct quantity

        if qty is None:
            raise ValueError("Order must have a positive quantity")

        if side == OrderSide.SELL:
            # Oversell and zero-quantity checks must come before any mutation
            if self.quantity == 0:
                raise ValueError("Cannot sell when position quantity is 0")
            if qty <= 0:
                raise ValueError("Order must have a positive quantity")
            if qty > self.quantity:
                raise ValueError("Cannot sell more than available quantity")
            # For sells, calculate realized PnL and update position
            self.realized_pnl += (executed_price - self.entry_price) * min(qty, self.quantity)
            if decrement:
                self.quantity = max(Decimal('0'), self.quantity - qty)
            # Reset position if fully closed
            if self.quantity == 0:
                self.reset()
            else:
                self.side = PositionSide.LONG
        elif side == OrderSide.BUY:
            if qty <= 0:
                raise ValueError("Order must have a positive quantity")
            # For buys, update entry price using weighted average
            if self.quantity == 0:
                self.entry_price = executed_price
                self.quantity = qty
            else:
                # Calculate weighted average entry price
                total_value = (self.quantity * self.entry_price) + (qty * executed_price)
                self.quantity += qty
                self.entry_price = total_value / self.quantity
            self.side = PositionSide.LONG
        # Update current price and unrealized PnL
        self.current_price = executed_price
        if self.quantity > 0:
            self.unrealized_pnl = (self.current_price - self.entry_price) * self.quantity
        else:
            self.unrealized_pnl = Decimal('0')
        from datetime import datetime, timezone
        self.last_update_time = datetime.now(timezone.utc)

    def __str__(self):
        """String representation of the position."""
        return (f"Position(symbol={self.symbol}, quantity={self.quantity}, "
                f"entry_price={self.entry_price}, side={self.side}, "
                f"realized_pnl={self.realized_pnl}, unrealized_pnl={self.unrealized_pnl})")

    def __repr__(self):
        """String representation of the position."""
        return self.__str__()
