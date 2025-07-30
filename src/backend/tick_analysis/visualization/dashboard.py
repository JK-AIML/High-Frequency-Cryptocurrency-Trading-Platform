"""Dashboard class for tick analysis visualization."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Optional, Union
import numpy as np

class Dashboard:
    """Interactive dashboard for visualizing trading data and metrics."""

    def __init__(self):
        """Initialize the dashboard."""
        self.fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3],
            subplot_titles=("Price", "Volume")
        )

    def add_candlestick_chart(self, df: pd.DataFrame, name: str = "Candlestick") -> None:
        """Add candlestick chart to the dashboard.
        
        Args:
            df: DataFrame with OHLCV data
            name: Name for the chart
        """
        self.fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name=name
            ),
            row=1,
            col=1
        )
        
        self.fig.update_layout(
            xaxis_rangeslider_visible=False,
            yaxis_title="Price",
            showlegend=True
        )

    def add_volume_chart(self, df: pd.DataFrame, name: str = "Volume") -> None:
        """Add volume chart to the dashboard.
        
        Args:
            df: DataFrame with volume data
            name: Name for the chart
        """
        colors = ['red' if row['close'] < row['open'] else 'green' 
                 for _, row in df.iterrows()]
        
        self.fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['volume'],
                name=name,
                marker_color=colors
            ),
            row=2,
            col=1
        )
        
        self.fig.update_layout(
            yaxis2_title="Volume",
            showlegend=True
        )

    def add_trade_markers(self, trades: List[Dict], price_data: pd.Series) -> None:
        """Add trade markers to the chart.
        
        Args:
            trades: List of trade dictionaries
            price_data: Price series for reference
        """
        for trade in trades:
            color = 'green' if trade['side'] == 'BUY' else 'red'
            marker = 'triangle-up' if trade['side'] == 'BUY' else 'triangle-down'
            
            self.fig.add_trace(
                go.Scatter(
                    x=[trade['timestamp']],
                    y=[trade['price']],
                    mode='markers',
                    marker=dict(
                        color=color,
                        size=12,
                        symbol=marker,
                        line=dict(width=2, color='white')
                    ),
                    name=f"{trade['side']} {trade['symbol']}",
                    text=f"Price: {trade['price']:.2f}<br>Quantity: {trade['quantity']:.4f}",
                    hoverinfo='text'
                ),
                row=1,
                col=1
            )

    def add_technical_indicators(self, df: pd.DataFrame, indicators: List[str]) -> None:
        """Add technical indicators to the chart.
        
        Args:
            df: DataFrame with price data
            indicators: List of indicator names to add
        """
        for indicator in indicators:
            if indicator.lower() == 'sma':
                for period in [20, 50, 200]:
                    sma = df['close'].rolling(window=period).mean()
                    self.fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=sma,
                            name=f"SMA {period}",
                            line=dict(width=1)
                        ),
                        row=1,
                        col=1
                    )
            
            elif indicator.lower() == 'rsi':
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                
                self.fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=rsi,
                        name="RSI",
                        line=dict(width=1)
                    ),
                    row=1,
                    col=1
                )
            
            elif indicator.lower() == 'macd':
                exp1 = df['close'].ewm(span=12, adjust=False).mean()
                exp2 = df['close'].ewm(span=26, adjust=False).mean()
                macd = exp1 - exp2
                signal = macd.ewm(span=9, adjust=False).mean()
                
                self.fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=macd,
                        name="MACD",
                        line=dict(width=1)
                    ),
                    row=1,
                    col=1
                )
                
                self.fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=signal,
                        name="Signal",
                        line=dict(width=1)
                    ),
                    row=1,
                    col=1
                )

    def add_performance_metrics(self, metrics: Dict[str, float], title: str = "Performance Metrics") -> None:
        """Add performance metrics table to the dashboard.
        
        Args:
            metrics: Dictionary of metric names and values
            title: Title for the metrics table
        """
        # Create table data
        table_data = []
        for name, value in metrics.items():
            if isinstance(value, float):
                value = f"{value:.2%}" if 'return' in name.lower() or 'ratio' in name.lower() else f"{value:.2f}"
            table_data.append([name.replace('_', ' ').title(), value])
        
        # Add table
        self.fig.add_trace(
            go.Table(
                header=dict(
                    values=['Metric', 'Value'],
                    fill_color='paleturquoise',
                    align='left'
                ),
                cells=dict(
                    values=list(zip(*table_data)),
                    fill_color='lavender',
                    align='left'
                )
            ),
            row=1,
            col=1
        )
        
        self.fig.update_layout(title=title)

    def show(self) -> None:
        """Display the dashboard."""
        self.fig.update_layout(
            height=800,
            width=1200,
            template='plotly_dark',
            xaxis_rangeslider_visible=False,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        self.fig.show()
