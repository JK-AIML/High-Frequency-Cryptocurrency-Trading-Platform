import numpy as np
import pandas as pd
import plotly.graph_objects as go

class PerformanceAnalytics:
    def __init__(self) -> None:
        self.fig = None

    def _calculate_metrics(self, returns: pd.Series) -> dict:
        returns = returns.dropna()
        n = len(returns)
        if n == 0:
            return {"cagr": 0, "sharpe": 0, "max_drawdown": 0, "win_rate": 0}
        periods_per_year = 252 if n > 50 else 12
        total_return = (returns + 1).prod() - 1
        years = n / periods_per_year
        cagr = (total_return + 1) ** (1 / years) - 1 if years > 0 else 0
        sharpe = returns.mean() / (returns.std() + 1e-9) * np.sqrt(periods_per_year)
        cum_returns = (returns + 1).cumprod()
        peak = cum_returns.cummax()
        drawdown = (cum_returns - peak) / peak
        max_drawdown = drawdown.min()
        win_rate = (returns > 0).sum() / n if n > 0 else 0
        return {
            "cagr": float(cagr),
            "sharpe": float(sharpe),
            "max_drawdown": float(max_drawdown),
            "win_rate": float(win_rate),
        }

    def render_equity_curve(self, returns: pd.Series, benchmark: pd.Series = None):
        equity = (returns + 1).cumprod()
        self.fig = go.Figure()
        self.fig.add_trace(go.Scatter(x=equity.index, y=equity, mode='lines', name='Strategy'))
        if benchmark is not None:
            bench_eq = (benchmark + 1).cumprod()
            self.fig.add_trace(go.Scatter(x=bench_eq.index, y=bench_eq, mode='lines', name='Benchmark'))
        self.fig.update_layout(title="Equity Curve", xaxis_title="Date", yaxis_title="Equity")
        return self.fig

    def render_drawdown(self, returns: pd.Series):
        cum_returns = (returns + 1).cumprod()
        peak = cum_returns.cummax()
        drawdown = (cum_returns - peak) / peak
        self.fig = go.Figure()
        self.fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown, mode='lines', name='Drawdown'))
        self.fig.update_layout(title="Drawdown", xaxis_title="Date", yaxis_title="Drawdown")
        return self.fig

    def render_returns_distribution(self, returns: pd.Series):
        self.fig = go.Figure()
        self.fig.add_trace(go.Histogram(x=returns, nbinsx=30, name='Returns'))
        self.fig.update_layout(title="Returns Distribution", xaxis_title="Returns", yaxis_title="Frequency")
        return self.fig

    def render_metrics(self, returns: pd.Series):
        metrics = self._calculate_metrics(returns)
        self.fig = go.Figure(data=[go.Table(
            header=dict(values=list(metrics.keys())),
            cells=dict(values=[[metrics[k]] for k in metrics])
        )])
        self.fig.update_layout(title="Performance Metrics")
        return self.fig
