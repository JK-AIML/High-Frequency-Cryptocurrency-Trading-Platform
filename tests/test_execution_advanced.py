"""Advanced tests for the execution module."""

import unittest
from decimal import Decimal
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

from tick_analysis.execution import (
    OrderStatus,
    Order,
    OrderSide,
    OrderType,
    TimeInForce,
    PositionSide,
    ExecutionHandler,
    InsufficientFunds,
    OrderRejected,
    Trade,
    InvalidOrder,
    MarketClosed,
    PositionLimitExceeded
)
from tick_analysis.portfolio import Portfolio


class TestAdvancedExecutionHandler(unittest.TestCase):
    """Advanced tests for the ExecutionHandler class."""

    def setUp(self):
        """Set up test fixtures."""
        self.initial_cash = Decimal("100000")
        self.portfolio = Portfolio(initial_cash=self.initial_cash)
        self.symbol = "BTC/USDT"
        self.price = Decimal("50000")
        self.quantity = Decimal("0.1")

        # Setup execution handler with test parameters
        self.executor = ExecutionHandler(
            portfolio=self.portfolio,
            slippage=0.0005,  # 0.05%
            fee_rate=0.001,  # 0.1%
            max_leverage=1.0,
            enable_shorting=True,
            dry_run=False,
        )

        # Initialize market data
        self._update_market_data()

    def _update_market_data(self, spread_pct=0.001):
        """Helper method to update market data with a given spread percentage."""
        spread = self.price * Decimal(str(spread_pct))
        self.executor.update_market_data(
            self.symbol,
            float(self.price - spread),  # bid
            float(self.price + spread),  # ask
        )

    def test_limit_order_execution(self):
        """Test limit order execution with price levels."""
        # Place a limit buy order below market
        limit_price = self.price * Decimal("0.99")  # 1% below market
        order = Order(
            symbol=self.symbol,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=self.quantity,
            limit_price=limit_price,
            time_in_force=TimeInForce.GTC,
        )

        # Submit the order
        submitted_order = self.executor.submit_order(order)
        self.assertEqual(submitted_order.status, OrderStatus.NEW)

        # Update market data to trigger limit order
        self.executor.update_market_data(self.symbol, limit_price, limit_price)
        self.executor.check_pending_orders()  # Call check_pending_orders to process orders

        # Check if order was filled
        filled_orders = self.executor.get_filled_orders()
        self.assertEqual(len(filled_orders), 1)
        self.assertEqual(filled_orders[0].status, OrderStatus.FILLED)

    def test_stop_order_triggering(self):
        """Test stop order triggering and execution."""
        # Place a stop buy order above market
        stop_price = self.price * Decimal("1.01")  # 1% above market
        order = Order(
            symbol=self.symbol,
            side=OrderSide.BUY,
            order_type=OrderType.STOP,
            quantity=self.quantity,
            stop_price=stop_price,
            time_in_force=TimeInForce.GTC,
        )

        # Submit the order
        submitted_order = self.executor.submit_order(order)
        self.assertEqual(submitted_order.status, OrderStatus.NEW)

        # Market moves up to trigger the stop
        self.price = stop_price * Decimal("1.001")  # Just above stop price
        self._update_market_data()
        self.executor.check_pending_orders()  # Check pending orders after updating market data

        # Check if order was triggered and filled
        filled_orders = self.executor.get_filled_orders()
        self.assertEqual(len(filled_orders), 1)
        self.assertEqual(filled_orders[0].status, OrderStatus.FILLED)

    @patch("tick_analysis.execution.handler.ExecutionHandler._execute_market_order")
    def test_insufficient_funds(self, mock_execute):
        """Test order rejection due to insufficient funds."""
        # Mock the execute method to raise InsufficientFundsError
        mock_execute.side_effect = InsufficientFunds("Not enough funds")

        # Try to place an order that's too large
        large_quantity = Decimal("1000")  # This would be 50M USDT at 50k/BTC
        order = Order(
            symbol=self.symbol,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=large_quantity,
        )

        # Should raise InsufficientFundsError
        with self.assertRaises(InsufficientFunds):
            self.executor.submit_order(order)

    def test_position_management(self):
        """Test position management with multiple orders."""
        # Initial buy
        buy_order = Order(
            symbol=self.symbol,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=self.quantity,
        )
        self.executor.submit_order(buy_order)

        # Check position
        position = self.portfolio.get_position(self.symbol)
        self.assertEqual(position.quantity, self.quantity)

        # Partial sell
        sell_quantity = self.quantity / 2
        sell_order = Order(
            symbol=self.symbol,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=sell_quantity,
        )
        self.executor.submit_order(sell_order)

        # Check updated position
        position = self.portfolio.get_position(self.symbol)
        self.assertEqual(position.quantity, self.quantity - sell_quantity)

    def test_trade_history(self):
        """Test trade history recording."""
        # Place and fill an order
        order = Order(
            symbol=self.symbol,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=self.quantity,
        )
        self.executor.submit_order(order)

        # Check trade history
        trades = self.executor.get_trade_history()
        self.assertGreaterEqual(len(trades), 1)
        self.assertIsInstance(trades[0], Trade)
        self.assertEqual(trades[0].symbol, self.symbol)
        self.assertEqual(trades[0].quantity, self.quantity)
        self.assertGreater(trades[0].price, 0)


if __name__ == "__main__":
    unittest.main()
