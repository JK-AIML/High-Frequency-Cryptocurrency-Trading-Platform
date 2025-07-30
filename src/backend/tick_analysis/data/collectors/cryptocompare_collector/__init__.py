from .requests import get
import sys
import tick_analysis.data.collectors.cryptocompare_collector.requests as requests

class CryptoCompareCollector:
    requests = requests
    pass

# Expose requests for patching compatibility in tests
sys.modules[__name__ + '.requests'] = requests
