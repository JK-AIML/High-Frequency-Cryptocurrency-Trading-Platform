"""Portfolio interfaces for tick analysis."""

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Dict, Optional

from tick_analysis.execution.order import Order, Trade

class IPortfolio(ABC):
    """Interface for portfolio management."""
    
    @abstractmethod
    def execute_order(self, order: Order, portfolio_cash: Optional[Decimal] = None) -> Order:
        """Execute an order and update portfolio state."""
        pass
    
    @abstractmethod
    def get_position(self, symbol: str):
        """Get position for a symbol."""
        pass
    
    @abstractmethod
    def update_equity(self) -> None:
        """Update portfolio equity."""
        pass
    
    @abstractmethod
    def close_position(self, symbol: str) -> bool:
        """Close a position."""
        pass
    
    @abstractmethod
    def get_summary(self) -> Dict:
        """Get portfolio summary."""
        pass 