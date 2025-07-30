import plotly.graph_objects as go
import numpy as np
import pandas as pd

class PortfolioAllocation:
    def __init__(self) -> None:
        self.fig = None

    def render_allocation_chart(self, weights):
        labels = list(weights.keys())
        values = list(weights.values())
        self.fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])
        self.fig.update_layout(title="Portfolio Allocation")
        return self.fig

    def render_risk_contribution(self, weights, cov_matrix):
        # Compute risk contributions
        w = np.array(list(weights.values()))
        cov = np.array(cov_matrix)
        portfolio_var = np.dot(w, np.dot(cov, w))
        marginal_contribs = np.dot(cov, w)
        risk_contribs = w * marginal_contribs / portfolio_var
        self.fig = go.Figure(data=[go.Bar(x=list(weights.keys()), y=risk_contribs)])
        self.fig.update_layout(title="Risk Contribution", xaxis_title="Asset", yaxis_title="Risk Contribution")
        return self.fig

    def render_efficient_frontier(self, expected_returns, cov_matrix, risk_free_rate=0.02):
        # Simple efficient frontier (random portfolios)
        n_assets = len(expected_returns)
        n_portfolios = 1000
        results = np.zeros((3, n_portfolios))
        for i in range(n_portfolios):
            weights = np.random.dirichlet(np.ones(n_assets), 1)[0]
            ret = np.dot(weights, expected_returns)
            vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe = (ret - risk_free_rate) / (vol + 1e-9)
            results[0,i] = vol
            results[1,i] = ret
            results[2,i] = sharpe
        self.fig = go.Figure(
            data=[go.Scatter(x=results[0], y=results[1], mode='markers', marker=dict(color=results[2], colorscale='Viridis', showscale=True), name='Portfolios')]
        )
        self.fig.update_layout(title="Efficient Frontier", xaxis_title="Volatility", yaxis_title="Return")
        return self.fig
