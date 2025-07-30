"""Order execution handler for tick analysis."""

from decimal import Decimal
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
import logging

from tick_analysis.execution.order import (
    Order,
    OrderStatus,
    OrderType,
    OrderSide,
    TimeInForce,
    PositionSide,
    Trade,
)
from tick_analysis.portfolio.interfaces import IPortfolio
from tick_analysis.portfolio.base import Portfolio
from tick_analysis.execution.exceptions import (
    ExecutionError,
    InsufficientFunds,
    OrderRejected,
    InvalidOrder,
    MarketClosed,
    PositionLimitExceeded
)
from tick_analysis.api.websocket import manager as websocket_manager
import asyncio
from .binance_adapter import BinanceAdapter

logger = logging.getLogger(__name__)

class ExecutionHandler:
    """Handles order execution and trade management."""

    def __init__(
        self,
        portfolio: Optional[IPortfolio] = None,
        slippage: float = 0.0005,
        fee_rate: float = 0.001,
        max_leverage: float = 1.0,
        enable_shorting: bool = False,
        dry_run: bool = False,
    ):
        """Initialize the execution handler."""
        self.portfolio = portfolio or Portfolio()
        self.slippage = Decimal(str(slippage))
        self.fee_rate = Decimal(str(fee_rate))
        self.max_leverage = Decimal(str(max_leverage))
        self.enable_shorting = enable_shorting
        self.dry_run = dry_run

        # Market data
        self.market_data: Dict[str, Dict[str, float]] = {}
        
        # Order tracking
        self.orders: Dict[str, Order] = {}
        self.trades: List[Trade] = []
        self.pending_orders: Dict[str, List[Order]] = {}

        self.exchanges = {
            'binance': BinanceAdapter(),
            # Add more exchanges here in the future
        }
        self.default_exchange = 'binance'

    def update_market_data(self, symbol: str, bid: float, ask: float) -> None:
        """Update market data for a symbol."""
        self.market_data[symbol] = {
            "bid": Decimal(str(bid)),
            "ask": Decimal(str(ask)),
            "last_update": datetime.now(timezone.utc),
        }

    def get_market_price(self, symbol: str, side: OrderSide) -> Optional[Decimal]:
        """Get the current market price for a symbol."""
        if symbol not in self.market_data:
            return None

        data = self.market_data[symbol]
        if side == OrderSide.BUY:
            return data["ask"]  # Buy at ask
        else:
            return data["bid"]  # Sell at bid

    def route_order(self, order: Order) -> str:
        """
        Smart order routing: select the best exchange for this order.
        For now, always use Binance, but structure for future multi-exchange support.
        In the future, compare price, liquidity, and fees across exchanges.
        """
        # Example: log routing decision
        logger.info(f"Routing order {order.id} for {order.symbol} to {self.default_exchange}")
        return self.default_exchange

    def submit_order(self, order: Order) -> Order:
        """Submit an order for execution with smart order routing."""
        # Validate order
        if not self._validate_order(order):
            raise InvalidOrder(f"Invalid order: {order}")
        # Check if we have sufficient funds
        if not self._check_funds(order):
            raise InsufficientFunds(f"Insufficient funds for order: {order}")
        # Smart order routing
        exchange_name = self.route_order(order)
        exchange = self.exchanges[exchange_name]
        # Process the order based on type
        if order.order_type == OrderType.MARKET:
            # Place order on exchange (live trading)
            if not self.dry_run:
                result = exchange.place_order(
                    symbol=order.symbol,
                    side=order.side.name,
                    quantity=float(order.quantity),
                    order_type=order.order_type.name,
                    price=float(order.price) if order.price else None
                )
                logger.info(f"Order {order.id} placed on {exchange_name}: {result}")
            return self._execute_market_order(order)
        elif order.order_type == OrderType.LIMIT:
            return self._submit_limit_order(order)
        elif order.order_type == OrderType.STOP:
            return self._submit_stop_order(order)
        elif order.order_type == OrderType.STOP_LIMIT:
            return self._submit_stop_limit_order(order)
        else:
            raise InvalidOrder(f"Unsupported order type: {order.order_type}")

    def _validate_order(self, order: Order) -> bool:
        """Validate an order."""
        if not order.symbol or not order.quantity or not order.side:
            return False

        if order.quantity <= 0:
            return False

        if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
            if not order.limit_price or order.limit_price <= 0:
                return False

        if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
            if not order.stop_price or order.stop_price <= 0:
                return False

        return True

    def _check_funds(self, order: Order) -> bool:
        """Check if we have sufficient funds for an order."""
        if self.dry_run:
            return True

        price = self.get_market_price(order.symbol, order.side)
        if not price:
            return False

        required_funds = order.quantity * price
        if order.side == OrderSide.BUY:
            return self.portfolio.cash >= required_funds
        else:
            position = self.portfolio.get_position(order.symbol)
            return position and position.quantity >= order.quantity

    def _execute_market_order(self, order: Order) -> Order:
        """Execute a market order."""
        price = self.get_market_price(order.symbol, order.side)
        if not price:
            raise OrderRejected(f"No market price available for {order.symbol}")

        # Apply slippage
        if order.side == OrderSide.BUY:
            executed_price = price * (1 + self.slippage)
        else:
            executed_price = price * (1 - self.slippage)

        # Execute the order
        order.executed_price = executed_price
        order.executed_quantity = order.quantity
        order.status = OrderStatus.FILLED
        order.filled_at = datetime.now(timezone.utc)

        # Update portfolio
        self.portfolio.execute_order(order)

        # Record trade
        trade = Trade(
            symbol=order.symbol,
            order_type=order.order_type,
            side=order.side,
            quantity=order.quantity,
            price=executed_price,
            timestamp=order.filled_at,
            order_id=order.id,
        )
        self.trades.append(trade)
        # WebSocket: broadcast trade update
        asyncio.create_task(websocket_manager.broadcast_trade(trade.__dict__))
        return order

    def _submit_limit_order(self, order: Order) -> Order:
        """Submit a limit order."""
        # Store the order
        self.orders[order.id] = order
        order.status = OrderStatus.NEW

        # Add to pending orders
        if order.symbol not in self.pending_orders:
            self.pending_orders[order.symbol] = []
        self.pending_orders[order.symbol].append(order)

        return order

    def _submit_stop_order(self, order: Order) -> Order:
        """Submit a stop order."""
        # Store the order
        self.orders[order.id] = order
        order.status = OrderStatus.NEW

        # Add to pending orders
        if order.symbol not in self.pending_orders:
            self.pending_orders[order.symbol] = []
        self.pending_orders[order.symbol].append(order)

        return order

    def _submit_stop_limit_order(self, order: Order) -> Order:
        """Submit a stop-limit order."""
        # Store the order
        self.orders[order.id] = order
        order.status = OrderStatus.NEW

        # Add to pending orders
        if order.symbol not in self.pending_orders:
            self.pending_orders[order.symbol] = []
        self.pending_orders[order.symbol].append(order)

        return order

    def check_pending_orders(self) -> None:
        """Check and process pending orders."""
        for symbol, orders in self.pending_orders.items():
            current_price = self.get_market_price(symbol, OrderSide.BUY)
            if not current_price:
                continue

            for order in orders[:]:  # Copy list to allow modification during iteration
                if order.status != OrderStatus.NEW:
                    continue

                if order.order_type == OrderType.LIMIT:
                    self._check_limit_order(order, current_price)
                elif order.order_type == OrderType.STOP:
                    self._check_stop_order(order, current_price)
                elif order.order_type == OrderType.STOP_LIMIT:
                    self._check_stop_limit_order(order, current_price)

    def _check_limit_order(self, order: Order, current_price: Decimal) -> None:
        """Check if a limit order should be executed."""
        if order.side == OrderSide.BUY and current_price <= order.limit_price:
            # Buy limit order: execute if price is at or below limit
            self._execute_market_order(order)
            self.pending_orders[order.symbol].remove(order)
        elif order.side == OrderSide.SELL and current_price >= order.limit_price:
            # Sell limit order: execute if price is at or above limit
            self._execute_market_order(order)
            self.pending_orders[order.symbol].remove(order)

    def _check_stop_order(self, order: Order, current_price: Decimal) -> None:
        """Check if a stop order should be triggered."""
        if order.side == OrderSide.BUY and current_price >= order.stop_price:
            # Buy stop order: trigger if price is at or above stop
            self._execute_market_order(order)
            self.pending_orders[order.symbol].remove(order)
        elif order.side == OrderSide.SELL and current_price <= order.stop_price:
            # Sell stop order: trigger if price is at or below stop
            self._execute_market_order(order)
            self.pending_orders[order.symbol].remove(order)

    def _check_stop_limit_order(self, order: Order, current_price: Decimal) -> None:
        """Check if a stop-limit order should be triggered."""
        if order.side == OrderSide.BUY:
            if current_price >= order.stop_price:
                # Stop price reached, check limit
                if current_price <= order.limit_price:
                    self._execute_market_order(order)
                    self.pending_orders[order.symbol].remove(order)
        else:  # SELL
            if current_price <= order.stop_price:
                # Stop price reached, check limit
                if current_price >= order.limit_price:
                    self._execute_market_order(order)
                    self.pending_orders[order.symbol].remove(order)

    def get_filled_orders(self) -> List[Order]:
        """Get all filled orders."""
        return [order for order in self.orders.values() if order.status == OrderStatus.FILLED]

    def get_trade_history(self) -> List[Trade]:
        """Get trade history."""
        return self.trades

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        if order_id not in self.orders:
            return False

        order = self.orders[order_id]
        if order.status != OrderStatus.NEW:
            return False

        order.status = OrderStatus.CANCELLED
        if order.symbol in self.pending_orders:
            self.pending_orders[order.symbol].remove(order)

        return True

