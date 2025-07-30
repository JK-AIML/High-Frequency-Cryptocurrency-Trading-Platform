from tick_analysis.core.strategies.ml_strategy import MLStrategy
from tick_analysis.backtest.advanced_engine import AdvancedBacktestEngine
from tick_analysis.portfolio.optimizer import PortfolioOptimizer
import mlflow

class MLStrategyIntegration:
    def __init__(self, config=None):
        if config is None:
            config = {}
        self.config = config
        self.strategy = MLStrategy(
            model_type=config.get("model_type", "random_forest"),
            model_params=config.get("model_params", {})
        )
        self.backtest_engine = AdvancedBacktestEngine()
        self.portfolio_optimizer = PortfolioOptimizer()

    def prepare_data(self, df):
        features = self.strategy.create_features(df)
        target = self.strategy.create_target(df)
        return features, target

    def train_strategy(self, df):
        return self.strategy.train(df)

    def run_backtest(self, df):
        # Prepare features and signals for the test's mocks
        features = self.strategy.create_features(df)
        signals = self.strategy.generate_signals(features)
        result = self.backtest_engine.run(features, signals)
        # Return as dict for test compatibility
        if result is not None:
            return {
                "returns": getattr(result, "returns", None),
                "positions": getattr(result, "positions", None),
                "trades": getattr(result, "trades", None),
                "metrics": getattr(result, "metrics", {}),
                "portfolio_values": getattr(result, "portfolio_values", []),
            }
        return {"metrics": {}, "portfolio_values": []}

    def optimize_portfolio(self, returns, signals=None):
        # This method is expected by the integration test
        return self.portfolio_optimizer.optimize_portfolio(returns, signals=signals)
