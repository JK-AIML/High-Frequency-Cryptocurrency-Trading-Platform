import os
import requests
from .exchange_interface import ExchangeInterface

class CryptoCompareAdapter(ExchangeInterface):
    """
    Market data provider using CryptoCompare API. Does NOT support trading or balances.
    """
    BASE_URL = "https://min-api.cryptocompare.com/data/"

    def __init__(self):
        self.api_key = os.getenv("CRYPTOCOMPARE_API_KEY")
        if not self.api_key:
            raise ValueError("CRYPTOCOMPARE_API_KEY not set in environment.")
        self.session = requests.Session()
        self.session.headers.update({"authorization": f"Apikey {self.api_key}"})

    def get_historical_data(self, symbol, start, end, timeframe):
        # timeframe: '1m', '5m', '1h', '1d', etc.
        import pandas as pd
        endpoint = f"histoday" if timeframe == "1d" else ("histohour" if timeframe.endswith("h") else "histominute")
        url = f"{self.BASE_URL}{endpoint}"
        params = {
            "fsym": symbol.split("/")[0],
            "tsym": symbol.split("/")[1] if "/" in symbol else "USD",
            "limit": 2000,  # max per request
            "toTs": int(pd.Timestamp(end).timestamp()),
        }
        resp = self.session.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()["Data"]
        df = pd.DataFrame(data)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        return df

    def get_realtime_price(self, symbol):
        url = f"{self.BASE_URL}price"
        params = {
            "fsym": symbol.split("/")[0],
            "tsyms": symbol.split("/")[1] if "/" in symbol else "USD",
        }
        resp = self.session.get(url, params=params)
        resp.raise_for_status()
        return resp.json()

    def place_order(self, *args, **kwargs):
        raise NotImplementedError("CryptoCompare is a market data provider only. Trading is not supported.")

    def get_balance(self, *args, **kwargs):
        raise NotImplementedError("CryptoCompare is a market data provider only. Balances are not supported.") 