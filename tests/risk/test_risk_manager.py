"""
Tests for the RiskManager class.
"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from decimal import Decimal

from tick_analysis.risk.manager import RiskManager
from tick_analysis.data.models import Candle, Timeframe, Order, OrderType, OrderSide, Position


class TestRiskManager(unittest.TestCase):
    """Test cases for RiskManager."""

    def setUp(self):
        """Set up test fixtures."""
        # Initialize test data
        self.symbol = "BTC/USDT"
        self.timeframe = Timeframe.ONE_HOUR
        self.initial_balance = Decimal("100000")

        # Create sample candles
        np.random.seed(42)
        n_candles = 1000
        base_price = 50000  # Base price in USDT

        # Generate random walk prices
        returns = np.random.normal(0.001, 0.02, n_candles)
        prices = base_price * (1 + np.cumsum(returns))

        self.candles = [
            Candle(
                symbol=self.symbol,
                timeframe=self.timeframe,
                timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc)
                + timedelta(hours=i),
                open=Decimal(str(prices[i])),
                high=Decimal(str(prices[i] * 1.01)),
                low=Decimal(str(prices[i] * 0.99)),
                close=Decimal(str(prices[i])),
                volume=Decimal(str(np.random.uniform(100, 1000))),
            )
            for i in range(n_candles)
        ]

        # Create sample positions
        self.positions = {
            self.symbol: Position(
                symbol=self.symbol,
                amount=Decimal("0.5"),  # 0.5 BTC
                entry_price=Decimal("50000"),
                current_price=Decimal("51000"),
                timestamp=datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            )
        }

        # Initialize risk manager
        self.risk_manager = RiskManager(
            max_position_size=Decimal("0.1"),  # 10% of portfolio
            max_leverage=Decimal("5.0"),
            stop_loss_pct=Decimal("0.05"),  # 5% stop loss
            take_profit_pct=Decimal("0.10"),  # 10% take profit
            max_drawdown=Decimal("0.20"),  # 20% max drawdown
            var_confidence=Decimal("0.95"),  # 95% VaR
            initial_balance=self.initial_balance,
        )

    def test_initialization(self):
        """Test RiskManager initialization."""
        self.assertEqual(self.risk_manager.max_position_size, Decimal("0.1"))
        self.assertEqual(self.risk_manager.max_leverage, Decimal("5.0"))
        self.assertEqual(self.risk_manager.stop_loss_pct, Decimal("0.05"))
        self.assertEqual(self.risk_manager.take_profit_pct, Decimal("0.10"))
        self.assertEqual(self.risk_manager.max_drawdown, Decimal("0.20"))
        self.assertEqual(self.risk_manager.var_confidence, Decimal("0.95"))
        self.assertEqual(self.risk_manager.initial_balance, self.initial_balance)

    def test_validate_order_size(self):
        """Test order size validation."""
        # Test valid order size
        valid_order = Order(
            order_id="1",
            symbol=self.symbol,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),  # 10% of portfolio at $50k = $5k position
            price=Decimal("50000"),
            timestamp=datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        )

        is_valid = self.risk_manager.validate_order_size(
            order=valid_order, current_balance=self.initial_balance
        )
        self.assertTrue(is_valid)

        # Test invalid order size (too large)
        invalid_order = Order(
            order_id="2",
            symbol=self.symbol,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("100"),  # $5M position (50x portfolio size)
            price=Decimal("50000"),
            timestamp=datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        )

        is_valid = self.risk_manager.validate_order_size(
            order=invalid_order, current_balance=self.initial_balance
        )
        self.assertFalse(is_valid)

    def test_check_leverage(self):
        """Test leverage validation."""
        # Test within leverage limit
        is_within_limit = self.risk_manager.check_leverage(
            position_size=Decimal("10000"),  # $10k position
            margin=Decimal("2000"),  # 5x leverage
        )
        self.assertTrue(is_within_limit)

        # Test exceeding leverage limit
        is_within_limit = self.risk_manager.check_leverage(
            position_size=Decimal("10000"),  # $10k position
            margin=Decimal("1000"),  # 10x leverage (over 5x limit)
        )
        self.assertFalse(is_within_limit)

    def test_calculate_var(self):
        """Test Value at Risk calculation."""
        # Create sample returns
        returns = pd.Series(np.random.normal(0.001, 0.02, 1000))

        # Calculate VaR
        var = self.risk_manager.calculate_var(returns=returns, confidence_level=0.95)

        # Check that VaR is negative (represents loss)
        self.assertLess(var, 0)

        # Check that VaR is within expected range for normal distribution
        self.assertGreater(
            var, -0.1
        )  # Shouldn't be more than 10% loss at 95% confidence

    def test_check_drawdown(self):
        """Test drawdown monitoring."""
        # Test within drawdown limit
        is_within_limit = self.risk_manager.check_drawdown(
            current_balance=Decimal("90000"),  # 10% drawdown
            peak_balance=Decimal("100000"),
        )
        self.assertTrue(is_within_limit)

        # Test exceeding drawdown limit
        is_within_limit = self.risk_manager.check_drawdown(
            current_balance=Decimal("70000"),  # 30% drawdown
            peak_balance=Decimal("100000"),
        )
        self.assertFalse(is_within_limit)

    def test_check_position_risk(self):
        """Test position risk assessment."""
        # Test position within risk limits
        position = self.positions[self.symbol]

        risk_assessment = self.risk_manager.check_position_risk(
            position=position, current_price=Decimal("49500")  # Slight loss
        )

        self.assertEqual(risk_assessment["status"], "OK")
        self.assertTrue(risk_assessment["is_within_limits"])

        # Test position hitting stop loss
        risk_assessment = self.risk_manager.check_position_risk(
            position=position,
            current_price=Decimal("45000"),  # 10% loss (beyond 5% stop loss)
        )

        self.assertEqual(risk_assessment["status"], "STOP_LOSS")
        self.assertFalse(risk_assessment["is_within_limits"])

        # Test position hitting take profit
        risk_assessment = self.risk_manager.check_position_risk(
            position=position,
            current_price=Decimal("55000"),  # 10% profit (hit take profit)
        )

        self.assertEqual(risk_assessment["status"], "TAKE_PROFIT")
        self.assertFalse(risk_assessment["is_within_limits"])


if __name__ == "__main__":
    unittest.main()
