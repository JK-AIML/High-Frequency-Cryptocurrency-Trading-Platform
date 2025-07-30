"""
Dashboard component for strategy optimization.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional
import json
from datetime import datetime, timedelta

from ...optimization.optimizer import StrategyOptimizer
from ...data import DataCollector
from ...utils.cache import AdvancedCache
from ...strategies import get_strategy_class

class OptimizationDashboard:
    """Interactive dashboard for strategy optimization."""
    
    def __init__(
        self,
        data_collector: DataCollector,
        cache: Optional[AdvancedCache] = None
    ):
        """
        Initialize the optimization dashboard.
        
        Args:
            data_collector: Data collector instance
            cache: Optional cache for storing optimization results
        """
        self.data_collector = data_collector
        self.cache = cache
        self.optimizer = None
        self.current_results = None
        
    def render(self):
        """Render the optimization dashboard."""
        st.title("Strategy Optimization")
        
        # Strategy selection
        strategy_name = st.selectbox(
            "Select Strategy",
            options=["MLStrategy", "VolatilityStrategy", "VolumeStrategy"]
        )
        
        # Get strategy class
        strategy_class = get_strategy_class(strategy_name)
        
        # Initialize optimizer
        self.optimizer = StrategyOptimizer(
            strategy_class=strategy_class,
            data_collector=self.data_collector,
            cache=self.cache
        )
        
        # Data selection
        col1, col2 = st.columns(2)
        with col1:
            symbol = st.text_input("Symbol", value="BTC/USD")
            timeframe = st.selectbox(
                "Timeframe",
                options=["1m", "5m", "15m", "1h", "4h", "1d"]
            )
        with col2:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=365)
            )
            end_date = st.date_input(
                "End Date",
                value=datetime.now()
            )
        
        # Parameter grid
        st.subheader("Parameter Grid")
        param_grid = self._render_parameter_grid(strategy_class)
        
        # Optimization settings
        st.subheader("Optimization Settings")
        col1, col2 = st.columns(2)
        with col1:
            method = st.selectbox(
                "Optimization Method",
                options=["grid", "random", "bayesian", "genetic"]
            )
            metric = st.selectbox(
                "Optimization Metric",
                options=["sharpe_ratio", "sortino_ratio", "calmar_ratio", "total_return"]
            )
        with col2:
            n_trials = st.number_input(
                "Number of Trials",
                min_value=10,
                max_value=1000,
                value=100
            )
            cv_splits = st.number_input(
                "Cross-Validation Splits",
                min_value=2,
                max_value=10,
                value=5
            )
        
        # Run optimization
        if st.button("Run Optimization"):
            with st.spinner("Running optimization..."):
                try:
                    # Run optimization
                    results = self.optimizer.optimize(
                        symbol=symbol,
                        timeframe=timeframe,
                        start_date=start_date,
                        end_date=end_date,
                        param_grid=param_grid,
                        method=method,
                        metric=metric,
                        n_trials=n_trials,
                        cv_splits=cv_splits
                    )
                    
                    # Store results
                    self.current_results = results
                    
                    # Display results
                    self._display_results(results)
                    
                except Exception as e:
                    st.error(f"Optimization failed: {str(e)}")
        
        # Display optimization history
        if self.optimizer.optimization_history:
            st.subheader("Optimization History")
            self._display_history()
    
    def _render_parameter_grid(self, strategy_class: type) -> Dict[str, List[Any]]:
        """Render parameter grid input fields."""
        param_grid = {}
        
        # Get strategy parameters
        strategy_params = strategy_class.get_parameters()
        
        # Create input fields for each parameter
        for param_name, param_info in strategy_params.items():
            param_type = param_info["type"]
            param_range = param_info["range"]
            
            if param_type == "int":
                values = st.multiselect(
                    f"{param_name} (int)",
                    options=range(param_range[0], param_range[1] + 1),
                    default=[param_range[0], param_range[1]]
                )
            elif param_type == "float":
                values = st.multiselect(
                    f"{param_name} (float)",
                    options=np.linspace(param_range[0], param_range[1], 10),
                    default=[param_range[0], param_range[1]]
                )
            else:
                values = st.multiselect(
                    f"{param_name} (categorical)",
                    options=param_range,
                    default=param_range
                )
            
            if values:
                param_grid[param_name] = values
        
        return param_grid
    
    def _display_results(self, results: Dict[str, Any]):
        """Display optimization results."""
        # Best parameters
        st.subheader("Best Parameters")
        st.json(results["best_params"])
        
        # Performance metrics
        st.subheader("Performance Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Best Score", f"{results['best_score']:.4f}")
        with col2:
            st.metric("Final Score", f"{results['final_score']:.4f}")
        with col3:
            if "std" in results:
                st.metric("Standard Deviation", f"{results['std']:.4f}")
        
        # Parameter importance
        if "study" in results:
            st.subheader("Parameter Importance")
            fig = go.Figure()
            importance = optuna.importance.get_param_importances(results["study"])
            fig.add_trace(go.Bar(
                x=list(importance.keys()),
                y=list(importance.values())
            ))
            fig.update_layout(
                title="Parameter Importance",
                xaxis_title="Parameter",
                yaxis_title="Importance"
            )
            st.plotly_chart(fig)
        
        # Optimization progress
        st.subheader("Optimization Progress")
        if "study" in results:
            # Optuna study
            trials = results["study"].trials
            scores = [t.value for t in trials if t.value is not None]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=scores,
                mode="lines+markers",
                name="Score"
            ))
            fig.update_layout(
                title="Optimization Progress",
                xaxis_title="Trial",
                yaxis_title="Score"
            )
            st.plotly_chart(fig)
        else:
            # Grid/Random search results
            scores = [r["score"] for r in results["results"]]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=scores,
                mode="lines+markers",
                name="Score"
            ))
            fig.update_layout(
                title="Optimization Progress",
                xaxis_title="Trial",
                yaxis_title="Score"
            )
            st.plotly_chart(fig)
        
        # Download results
        st.download_button(
            "Download Results",
            data=json.dumps(results, indent=2),
            file_name=f"optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    def _display_history(self):
        """Display optimization history."""
        history_df = pd.DataFrame(self.optimizer.optimization_history)
        
        # Convert timestamp to datetime
        history_df["timestamp"] = pd.to_datetime(history_df["timestamp"])
        
        # Display history table
        st.dataframe(
            history_df[["timestamp", "method", "metric", "best_score"]]
        )
        
        # Plot history
        fig = go.Figure()
        for method in history_df["method"].unique():
            method_data = history_df[history_df["method"] == method]
            fig.add_trace(go.Scatter(
                x=method_data["timestamp"],
                y=method_data["best_score"],
                mode="lines+markers",
                name=method
            ))
        fig.update_layout(
            title="Optimization History",
            xaxis_title="Timestamp",
            yaxis_title="Best Score"
        )
        st.plotly_chart(fig) 