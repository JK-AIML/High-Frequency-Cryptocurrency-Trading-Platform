class RiskMetrics:
    def __init__(self) -> None:
        pass

import plotly.graph_objects as go
import numpy as np
import pandas as pd

class RiskMetricsDisplay:
    def __init__(self) -> None:
        self.fig = None

    def render_var_cvar(self, returns, confidence_level=0.95):
        returns = returns.dropna()
        var = np.percentile(returns, 100 * (1 - confidence_level))
        cvar = returns[returns <= var].mean()
        self.fig = go.Figure()
        self.fig.add_trace(go.Histogram(x=returns, nbinsx=30, name='Returns'))
        self.fig.add_vline(x=var, line_color='red', annotation_text=f'VaR: {var:.4f}')
        self.fig.add_vline(x=cvar, line_color='orange', annotation_text=f'CVaR: {cvar:.4f}')
        self.fig.update_layout(title="VaR and CVaR", xaxis_title="Returns", yaxis_title="Frequency")
        return self.fig

    def render_drawdown_analysis(self, returns):
        cum_returns = (returns + 1).cumprod()
        peak = cum_returns.cummax()
        drawdown = (cum_returns - peak) / peak
        self.fig = go.Figure()
        self.fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown, mode='lines', name='Drawdown'))
        self.fig.update_layout(title="Drawdown Analysis", xaxis_title="Date", yaxis_title="Drawdown")
        return self.fig

    def render_correlation_heatmap(self, returns_df):
        corr = returns_df.corr()
        self.fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, colorscale='Viridis'))
        self.fig.update_layout(title="Correlation Heatmap")
        return self.fig
