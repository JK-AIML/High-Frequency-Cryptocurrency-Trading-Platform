from .engine import BacktestEngine

class BacktestConfig:
    def __init__(self, initial_capital=100000.0, commission=0.001, slippage=0.0, **kwargs) -> None:
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        for k, v in kwargs.items():
            setattr(self, k, v)

class BacktestResult:
    def __init__(self, returns=None, positions=None, trades=None, portfolio_values=None, metrics=None, **kwargs):
        self.returns = returns
        self.positions = positions
        self.trades = trades
        self.portfolio_values = portfolio_values
        self.metrics = metrics or {}
        for k, v in kwargs.items():
            setattr(self, k, v)

class AdvancedBacktestEngine:
    def __init__(self, config: BacktestConfig = None) -> None:
        self.config = config or BacktestConfig()
        self.engine = BacktestEngine(
            initial_capital=self.config.initial_capital,
            commission=self.config.commission,
            slippage=self.config.slippage
        )

    def run(self, data=None, generate_signals=None, strategy_params=None, **kwargs):
        # generate_signals should be a function(engine, data) -> (entries, exits)
        results = self.engine.run(
            strategy=generate_signals,
            data=data,
            **kwargs
        )
        metrics = self.engine.calculate_risk_metrics(results.get("returns"))
        return BacktestResult(
            returns=results.get("returns"),
            positions=results.get("positions"),
            trades=results.get("trades"),
            portfolio_values=results.get("portfolio_values"),
            metrics=metrics
        )

class AdvancedEngine:
    def __init__(self) -> None:
        pass
