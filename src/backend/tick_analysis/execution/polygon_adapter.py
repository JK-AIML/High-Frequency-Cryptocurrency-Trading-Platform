import os
import requests
from .exchange_interface import ExchangeInterface
import pandas as pd

class PolygonAdapter(ExchangeInterface):
    """
    Market data provider using Polygon API. Does NOT support trading or balances in this implementation.
    """
    BASE_URL = "https://api.polygon.io/"

    def __init__(self):
        self.api_key = os.getenv("POLYGON_API_KEY")
        if not self.api_key:
            raise ValueError("POLYGON_API_KEY not set in environment.")
        self.session = requests.Session()

    def get_historical_data(self, symbol, start, end, timeframe):
        # timeframe: 'minute', 'hour', 'day', etc.
        url = f"{self.BASE_URL}v2/aggs/ticker/{symbol}/range/1/{timeframe}/{start}/{end}"
        params = {"apiKey": self.api_key}
        resp = self.session.get(url, params=params)
        resp.raise_for_status()
        data = resp.json().get("results", [])
        df = pd.DataFrame(data)
        if not df.empty:
            df["t"] = pd.to_datetime(df["t"], unit="ms")
            df.set_index("t", inplace=True)
        return df

    def get_realtime_price(self, symbol):
        url = f"{self.BASE_URL}v2/last/trade/{symbol}"
        params = {"apiKey": self.api_key}
        resp = self.session.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        return data.get("last", {}).get("price")

    def place_order(self, *args, **kwargs):
        raise NotImplementedError("Polygon is a market data provider only in this implementation. Trading is not supported.")
        # To enable trading, uncomment and implement the code below:
        # ...

    def get_balance(self, *args, **kwargs):
        raise NotImplementedError("Polygon is a market data provider only in this implementation. Balances are not supported.")
        # To enable trading, uncomment and implement the code below:
        # ... 