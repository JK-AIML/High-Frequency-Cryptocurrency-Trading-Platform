"""
Custom exceptions for the execution module.
"""

class ExecutionError(Exception):
    """Base exception for execution errors."""
    pass

class OrderRejected(ExecutionError):
    """Raised when an order is rejected by the exchange or broker."""
    pass

class InsufficientFunds(ExecutionError):
    """Raised when there are not enough funds to execute an order."""
    pass

class InvalidOrder(ExecutionError):
    """Raised when an order is invalid."""
    pass

class MarketClosed(ExecutionError):
    """Raised when trying to execute an order while the market is closed."""
    pass

class PositionLimitExceeded(ExecutionError):
    """Raised when a position limit is exceeded."""
    pass 