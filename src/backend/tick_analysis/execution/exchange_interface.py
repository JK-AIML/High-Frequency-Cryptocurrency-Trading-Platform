from abc import ABC, abstractmethod

class ExchangeInterface(ABC):
    @abstractmethod
    def get_historical_data(self, symbol, start, end, timeframe):
        """Fetch historical OHLCV or tick data for a symbol."""
        pass

    @abstractmethod
    def get_realtime_price(self, symbol):
        """Fetch the latest price for a symbol."""
        pass

    @abstractmethod
    def place_order(self, symbol, side, quantity, order_type, price=None):
        """Place an order (if supported)."""
        pass

    @abstractmethod
    def get_balance(self, asset):
        """Get balance for a specific asset (if supported)."""
        pass 