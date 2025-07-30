import plotly.graph_objects as go

class MarketVisualization:
    def __init__(self) -> None:
        self.fig = None

    def render_price_chart(self, data):
        self.fig = go.Figure()
        self.fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name='Price'
            )
        )
        self.fig.update_layout(title="Market Price Chart", xaxis_title="Date", yaxis_title="Price")
        return self.fig

    def render_technical_indicators(self, data, indicators):
        if self.fig is None:
            self.render_price_chart(data)
        for ind, props in indicators.items():
            if ind in data.columns:
                self.fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data[ind],
                        mode='lines',
                        name=ind,
                        line=dict(color=props.get('color', 'black'), dash=props.get('dash', 'solid'))
                    )
                )
        self.fig.update_layout(title="Market Price Chart with Indicators")
        return self.fig
