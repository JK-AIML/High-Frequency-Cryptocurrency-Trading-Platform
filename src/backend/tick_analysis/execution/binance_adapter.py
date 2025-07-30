import os
from .exchange_interface import ExchangeInterface
from binance.client import Client
import pandas as pd

class BinanceAdapter(ExchangeInterface):
    def __init__(self):
        self.api_key = os.getenv("BINANCE_API_KEY")
        self.api_secret = os.getenv("BINANCE_API_SECRET")
        if not self.api_key or not self.api_secret:
            raise ValueError("Binance API keys not set in environment.")
        self.client = Client(self.api_key, self.api_secret)

    def get_historical_data(self, symbol, start, end, timeframe):
        # symbol: e.g. 'BTCUSDT', timeframe: '1m', '1h', etc.
        klines = self.client.get_historical_klines(symbol, timeframe, start, end)
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df.set_index('open_time', inplace=True)
        df = df.astype(float, errors='ignore')
        return df

    def get_realtime_price(self, symbol):
        ticker = self.client.get_symbol_ticker(symbol=symbol)
        return float(ticker['price'])

    def place_order(self, symbol, side, quantity, order_type, price=None):
        # order_type: 'MARKET' or 'LIMIT'
        params = {
            'symbol': symbol,
            'side': side.upper(),
            'type': order_type.upper(),
            'quantity': quantity,
        }
        if order_type.upper() == 'LIMIT' and price is not None:
            params['price'] = price
            params['timeInForce'] = 'GTC'
        order = self.client.create_order(**params)
        return order

    def get_balance(self, asset):
        info = self.client.get_asset_balance(asset=asset)
        return info 