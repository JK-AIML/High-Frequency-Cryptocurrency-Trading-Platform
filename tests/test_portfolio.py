"""Tests for the portfolio module."""

import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import pandas as pd
import numpy as np
import pytest

from tick_analysis.portfolio import Portfolio, Position
from tick_analysis.execution import Order, OrderSide, OrderType, PositionSide, Trade
from tick_analysis.portfolio.interfaces import IPortfolio


class TestPosition(unittest.TestCase):
    """Test the Position class."""

    def setUp(self):
        """Set up test fixtures."""
        self.symbol = "AAPL"
        self.position = Position(symbol=self.symbol)

    def test_initialization(self):
        """Test position initialization."""
        self.assertEqual(self.position.symbol, self.symbol)
        self.assertEqual(self.position.quantity, Decimal("0"))
        self.assertEqual(self.position.entry_price, Decimal("0"))
        self.assertEqual(self.position.side, PositionSide.LONG)

    def test_update_market_value_long(self):
        """Test updating market value for a long position."""
        self.position.quantity = Decimal("10")
        self.position.entry_price = Decimal("100")
        self.position.side = PositionSide.LONG

        # Price increases
        self.position.update_market_value(110)
        self.assertEqual(self.position.current_price, Decimal("110"))
        self.assertEqual(
            self.position.unrealized_pnl, Decimal("100")
        )  # 10 * (110 - 100)

        # Price decreases
        self.position.update_market_value(105)
        self.assertEqual(
            self.position.unrealized_pnl, Decimal("50")
        )  # 10 * (105 - 100)

    def test_update_market_value_short(self):
        """Test updating market value for a short position."""
        self.position.quantity = Decimal("10")
        self.position.entry_price = Decimal("100")
        self.position.side = PositionSide.SHORT

        # Price decreases (profitable for short)
        self.position.update_market_value(90)
        self.assertEqual(
            self.position.unrealized_pnl, Decimal("100")
        )  # 10 * (100 - 90)

        # Price increases (loss for short)
        self.position.update_market_value(95)
        self.assertEqual(self.position.unrealized_pnl, Decimal("50"))  # 10 * (100 - 95)

    def test_add_trade_buy(self):
        """Test adding a buy trade to the position."""
        order = Order(
            symbol=self.symbol,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("10"),
            price=Decimal("100"),
            timestamp=datetime.now(timezone.utc),
        )
        order.executed_price = Decimal("100")
        order.executed_quantity = Decimal("10")

        self.position.add_trade(order)

        self.assertEqual(self.position.quantity, Decimal("10"))
        self.assertEqual(self.position.entry_price, Decimal("100"))
        self.assertEqual(self.position.side, PositionSide.LONG)

    def test_add_trade_sell(self):
        """Test adding a sell trade to the position."""
        # First, buy some shares
        buy_order = Order(
            symbol=self.symbol,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("10"),
            price=Decimal("100"),
            timestamp=datetime.now(timezone.utc),
        )
        buy_order.executed_price = Decimal("100")
        buy_order.executed_quantity = Decimal("10")
        self.position.add_trade(buy_order)

        # Now sell half
        sell_order = Order(
            symbol=self.symbol,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("5"),
            price=Decimal("110"),
            timestamp=datetime.now(timezone.utc),
        )
        sell_order.executed_price = Decimal("110")
        sell_order.executed_quantity = Decimal("5")

        self.position.add_trade(sell_order)

        self.assertEqual(self.position.quantity, Decimal("5"))
        self.assertEqual(
            self.position.entry_price, Decimal("100")
        )  # Avg price remains the same
        self.assertEqual(self.position.realized_pnl, Decimal("50"))  # 5 * (110 - 100)

    def test_position_edge_cases(self):
        """Test edge cases in position management."""
        # Create a position with valid initial state
        position = Position(
            symbol="AAPL",
            quantity=Decimal('100'),
            entry_price=Decimal('150.00'),
            side=PositionSide.LONG
        )
        
        # Test updating market value with None
        position.update_market_value(None)
        assert position.current_price == Decimal('150.00')
        assert position.unrealized_pnl == Decimal('0')
        
        # Test updating market value with negative price
        with pytest.raises(ValueError, match="Price cannot be negative"):
            position.update_market_value(Decimal('-150.00'))
            
        # Test adding trade with None quantity (expect error at Order creation)
        with pytest.raises(ValueError, match="Order must have a positive quantity"):
            Order(
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=None,
                price=Decimal('150.00'),
                order_type=OrderType.MARKET
            )
        
        # Test selling more than available quantity
        order = Order(
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=Decimal('200'),
            price=Decimal('150.00'),
            order_type=OrderType.MARKET
        )
        with pytest.raises(ValueError, match="Cannot sell more than available quantity"):
            position.add_trade(order)
        
        # Test selling when position quantity is 0
        position = Position(
            symbol="AAPL",
            quantity=Decimal('0'),
            entry_price=Decimal('150.00'),
            side=PositionSide.LONG
        )
        order = Order(
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=Decimal('10'),
            price=Decimal('150.00'),
            order_type=OrderType.MARKET
        )
        with pytest.raises(ValueError, match="Cannot sell when position quantity is 0"):
            position.add_trade(order)
        
        # Test position side transition
        position = Position(
            symbol="AAPL",
            quantity=Decimal('100'),
            entry_price=Decimal('150.00'),
            side=PositionSide.LONG
        )
        order = Order(
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=Decimal('100'),
            price=Decimal('150.00'),
            order_type=OrderType.MARKET
        )
        position.add_trade(order)
        assert position.quantity == Decimal('0')
        assert position.side == PositionSide.LONG  # Side should remain unchanged


class TestPortfolio(unittest.TestCase):
    """Test the Portfolio class."""

    def setUp(self):
        """Set up test fixtures."""
        self.initial_cash = Decimal("100000")
        self.portfolio = Portfolio(initial_cash=self.initial_cash)

    def test_initialization(self):
        """Test portfolio initialization."""
        self.assertEqual(self.portfolio.cash, self.initial_cash)
        self.assertEqual(self.portfolio.equity, self.initial_cash)
        self.assertEqual(len(self.portfolio.positions), 0)

    def test_execute_order_buy(self):
        """Test executing a buy order."""
        symbol = "AAPL"
        quantity = Decimal("10")
        price = Decimal("150")

        order = Order(
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=quantity,
            price=price,
            timestamp=datetime.now(timezone.utc),
        )
        order.executed_price = price
        order.executed_quantity = quantity

        self.portfolio.execute_order(order)

        # Check portfolio state - account for slippage and fees
        slippage = price * self.portfolio.slippage
        executed_price = price * (1 + slippage)  # Buy order adds slippage
        fee = (quantity * executed_price) * self.portfolio.fee_rate
        expected_cash = self.initial_cash - (quantity * executed_price) - fee
        self.assertAlmostEqual(
            float(self.portfolio.cash), float(expected_cash), places=4
        )
        self.assertIn(symbol, self.portfolio.positions)
        position = self.portfolio.positions[symbol]
        self.assertEqual(position.quantity, quantity)
        # Entry price should include slippage for market orders
        self.assertEqual(position.entry_price, executed_price)

    def test_execute_order_sell(self):
        """Test executing a sell order."""
        symbol = "AAPL"
        buy_quantity = Decimal("10")
        sell_quantity = Decimal("5")
        buy_price = Decimal("100")
        sell_price = Decimal("110")

        # First, buy some shares
        buy_order = Order(
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=buy_quantity,
            price=buy_price,
            timestamp=datetime.now(timezone.utc),
        )
        buy_order.executed_price = buy_price
        buy_order.executed_quantity = buy_quantity
        self.portfolio.execute_order(buy_order)

        # Now sell half
        sell_order = Order(
            symbol=symbol,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=sell_quantity,
            price=sell_price,
            timestamp=datetime.now(timezone.utc),
        )
        sell_order.executed_price = sell_price
        sell_order.executed_quantity = sell_quantity

        self.portfolio.execute_order(sell_order)

        # Check portfolio state - account for slippage and fees on both buy and sell
        # Buy order
        buy_slippage = buy_price * self.portfolio.slippage
        buy_executed_price = buy_price * (1 + buy_slippage)  # Buy order adds slippage
        buy_fee = (buy_quantity * buy_executed_price) * self.portfolio.fee_rate

        # Sell order
        sell_slippage = sell_price * self.portfolio.slippage
        sell_executed_price = sell_price * (
            1 - sell_slippage
        )  # Sell order subtracts slippage
        sell_fee = (sell_quantity * sell_executed_price) * self.portfolio.fee_rate

        expected_cash = (
            self.initial_cash - (buy_quantity * buy_executed_price) - buy_fee
        )
        expected_cash += (sell_quantity * sell_executed_price) - sell_fee

        self.assertAlmostEqual(
            float(self.portfolio.cash), float(expected_cash), places=4
        )

        position = self.portfolio.positions[symbol]
        self.assertEqual(position.quantity, buy_quantity - sell_quantity)

        # Calculate expected P&L with slippage and fees
        buy_slippage = buy_price * self.portfolio.slippage
        buy_executed_price = buy_price * (1 + buy_slippage)  # Buy order adds slippage

        sell_slippage = sell_price * self.portfolio.slippage
        sell_executed_price = sell_price * (
            1 - sell_slippage
        )  # Sell order subtracts slippage

        expected_pnl = (sell_executed_price - buy_executed_price) * sell_quantity
        self.assertAlmostEqual(
            float(position.realized_pnl), float(expected_pnl), places=4
        )

    def test_close_position(self):
        """Test closing a position."""
        symbol = "AAPL"
        quantity = Decimal("10")
        buy_price = Decimal("100")
        sell_price = Decimal("110")

        # Buy some shares
        buy_order = Order(
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=quantity,
            price=buy_price,
            timestamp=datetime.now(timezone.utc),
        )
        buy_order.executed_price = buy_price
        buy_order.executed_quantity = quantity
        self.portfolio.execute_order(buy_order)

        # Update market price
        position = self.portfolio.get_position(symbol)
        position.update_market_value(sell_price)

        # Close position
        self.portfolio.close_position(symbol)

        # Check that position is closed
        self.assertEqual(position.quantity, Decimal("0"))

        # Calculate expected P&L with slippage on buy and close
        buy_slippage = buy_price * self.portfolio.slippage
        buy_executed_price = buy_price * (1 + buy_slippage)  # Buy order adds slippage

        # For closing, we use the current price which was set to sell_price
        # The close order will also have slippage
        sell_slippage = sell_price * self.portfolio.slippage
        sell_executed_price = sell_price * (
            1 - sell_slippage
        )  # Sell order subtracts slippage

        expected_pnl = (sell_executed_price - buy_executed_price) * quantity
        self.assertAlmostEqual(
            float(position.realized_pnl), float(expected_pnl), places=4
        )

    def test_get_summary(self):
        """Test getting portfolio summary."""
        print("DEBUG: test_get_summary")
        for i, t in enumerate(self.portfolio.trade_history):
            print(f"TRADE {i}: {vars(t)}")
        summary = self.portfolio.get_summary()
        self.assertEqual(summary["initial_cash"], float(self.initial_cash))
        self.assertEqual(summary["current_equity"], float(self.initial_cash))
        self.assertEqual(summary["total_return_pct"], 0.0)

        # Make a profitable trade
        symbol = "AAPL"
        quantity = Decimal("10")
        buy_price = Decimal("100")
        sell_price = Decimal("110")

        # Buy
        buy_order = Order(
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=quantity,
            price=buy_price,
            timestamp=datetime.now(timezone.utc),
        )
        buy_order.executed_price = buy_price
        buy_order.executed_quantity = quantity
        self.portfolio.execute_order(buy_order)

        # Sell at higher price
        sell_order = Order(
            symbol=symbol,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=quantity,
            price=sell_price,
            timestamp=datetime.now(timezone.utc),
        )
        sell_order.executed_price = sell_price
        sell_order.executed_quantity = quantity
        self.portfolio.execute_order(sell_order)

        # Check summary
        summary = self.portfolio.get_summary()

        # Calculate expected P&L with slippage and fees
        buy_slippage = buy_price * self.portfolio.slippage
        buy_executed_price = buy_price * (1 + buy_slippage)  # Buy order adds slippage
        buy_fee = (quantity * buy_executed_price) * self.portfolio.fee_rate

        sell_slippage = sell_price * self.portfolio.slippage
        sell_executed_price = sell_price * (
            1 - sell_slippage
        )  # Sell order subtracts slippage
        sell_fee = (quantity * sell_executed_price) * self.portfolio.fee_rate

        # Total P&L from the trade
        total_pnl = (
            (sell_executed_price - buy_executed_price) * quantity - buy_fee - sell_fee
        )
        expected_return_pct = (float(total_pnl) / float(self.initial_cash)) * 100

        self.assertAlmostEqual(
            summary["total_return_pct"], expected_return_pct, places=2
        )
        self.assertEqual(summary["total_trades"], 2)
        self.assertEqual(summary["win_rate"], 100.0)  # Only one trade, which was a win

    def test_trade_history_debug(self):
        """Test trade history debugging functionality."""
        # Create a portfolio with initial cash
        portfolio = Portfolio(initial_cash=Decimal('10000'))
        
        # Execute a buy order
        buy_order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal('100'),
            price=Decimal('150.00'),
            order_type=OrderType.MARKET
        )
        portfolio.execute_order(buy_order)
        # Update market value to ensure position is updated
        portfolio.positions["AAPL"].update_market_value(Decimal('150.00'))
        
        # Execute a sell order
        sell_order = Order(
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=Decimal('50'),
            price=Decimal('160.00'),
            order_type=OrderType.MARKET
        )
        portfolio.execute_order(sell_order)
        
        # Verify trade history
        assert len(portfolio.trade_history) == 2
        assert portfolio.trade_history[0].symbol == "AAPL"
        assert portfolio.trade_history[0].side == OrderSide.BUY
        assert portfolio.trade_history[0].quantity == Decimal('100')
        assert portfolio.trade_history[0].price == Decimal('150.00')
        
        assert portfolio.trade_history[1].symbol == "AAPL"
        assert portfolio.trade_history[1].side == OrderSide.SELL
        assert portfolio.trade_history[1].quantity == Decimal('50')
        assert portfolio.trade_history[1].price == Decimal('160.00')
        assert portfolio.trade_history[1].realized_pnl == Decimal('500.00')  # (160 - 150) * 50

    def test_execute_order_edge_cases(self):
        """Test edge cases in order execution."""
        portfolio = Portfolio(initial_cash=Decimal('10000'))
        
        # Test zero quantity order
        with pytest.raises(ValueError, match="Order must have a positive quantity"):
            Order(
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=Decimal('0'),
                price=Decimal('150.00'),
                order_type=OrderType.MARKET
            )
        
        # Test negative quantity order
        with pytest.raises(ValueError, match="Order must have a positive quantity"):
            Order(
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=Decimal('-100'),
                price=Decimal('150.00'),
                order_type=OrderType.MARKET
            )
        
        # Test market order with None price
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal('100'),
            price=None,
            order_type=OrderType.MARKET
        )
        portfolio.execute_order(order)
        assert portfolio.positions["AAPL"].quantity == Decimal('100')
        assert portfolio.positions["AAPL"].entry_price == Decimal('150.00')
        
        # Test limit order with None price (expect error at Order creation)
        with pytest.raises(ValueError, match="Limit orders must have a limit price"):
            Order(
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=Decimal('100'),
                price=None,
                order_type=OrderType.LIMIT
            )

    def test_trade_history_edge_cases(self):
        """Test edge cases in trade history management."""
        # Test trade comparison with invalid type
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal('100'),
            price=Decimal('150.00'),
            order_type=OrderType.MARKET
        )
        trade = Trade(order)
        with pytest.raises(TypeError):
            trade == "invalid"
        
        # Test trade creation with missing required fields
        with pytest.raises(ValueError):
            Trade(None)
        
        # Test realized_pnl attribute
        assert hasattr(trade, 'realized_pnl')
        assert isinstance(trade.realized_pnl, Decimal)


if __name__ == "__main__":
    unittest.main()
