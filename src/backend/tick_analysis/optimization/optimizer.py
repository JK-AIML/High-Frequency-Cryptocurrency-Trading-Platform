"""
Strategy optimization framework for finding optimal parameters and configurations.
"""

from typing import Dict, List, Optional, Union, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from concurrent.futures import ProcessPoolExecutor
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import logging
from datetime import datetime, timedelta

from ..utils.cache import AdvancedCache
from ..utils.logging import setup_logger
from ..data import DataCollector
from ..strategies import BaseStrategy

logger = setup_logger(__name__)

class StrategyOptimizer:
    """Advanced strategy optimization framework with multiple optimization methods."""
    
    def __init__(
        self,
        strategy_class: type,
        data_collector: DataCollector,
        cache: Optional[AdvancedCache] = None,
        n_jobs: int = -1,
        random_state: int = 42
    ):
        """
        Initialize the strategy optimizer.
        
        Args:
            strategy_class: The strategy class to optimize
            data_collector: Data collector instance
            cache: Optional cache for storing optimization results
            n_jobs: Number of parallel jobs (-1 for all cores)
            random_state: Random seed for reproducibility
        """
        self.strategy_class = strategy_class
        self.data_collector = data_collector
        self.cache = cache
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.optimization_history = []
        
    def optimize(
        self,
        symbol: str,
        timeframe: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        param_grid: Dict[str, List[Any]],
        method: str = "grid",
        metric: str = "sharpe_ratio",
        n_trials: int = 100,
        cv_splits: int = 5,
        validation_size: float = 0.2,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Optimize strategy parameters using specified method.
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            start_date: Start date for optimization
            end_date: End date for optimization
            param_grid: Parameter grid for optimization
            method: Optimization method ("grid", "random", "bayesian", "genetic")
            metric: Optimization metric
            n_trials: Number of optimization trials
            cv_splits: Number of cross-validation splits
            validation_size: Size of validation set
            **kwargs: Additional optimization parameters
            
        Returns:
            Dictionary containing optimization results
        """
        # Convert dates to datetime
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
            
        # Get data
        data = self._get_optimization_data(
            symbol, timeframe, start_date, end_date
        )
        
        # Split data
        train_data, val_data = self._split_data(data, validation_size)
        
        # Select optimization method
        if method == "grid":
            results = self._grid_search(
                train_data, val_data, param_grid, metric, cv_splits
            )
        elif method == "random":
            results = self._random_search(
                train_data, val_data, param_grid, metric, n_trials, cv_splits
            )
        elif method == "bayesian":
            results = self._bayesian_optimization(
                train_data, val_data, param_grid, metric, n_trials, cv_splits
            )
        elif method == "genetic":
            results = self._genetic_optimization(
                train_data, val_data, param_grid, metric, n_trials, cv_splits
            )
        else:
            raise ValueError(f"Unknown optimization method: {method}")
            
        # Store optimization history
        self.optimization_history.append({
            "timestamp": datetime.now(),
            "method": method,
            "metric": metric,
            "results": results
        })
        
        return results
    
    def _get_optimization_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Get and prepare data for optimization."""
        # Check cache first
        if self.cache:
            cache_key = f"opt_data_{symbol}_{timeframe}_{start_date}_{end_date}"
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                return cached_data
                
        # Get data from collector
        data = self.data_collector.get_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_date,
            end_time=end_date
        )
        
        # Cache data
        if self.cache:
            self.cache.set(cache_key, data)
            
        return data
    
    def _split_data(
        self,
        data: pd.DataFrame,
        validation_size: float
    ) -> tuple:
        """Split data into training and validation sets."""
        split_idx = int(len(data) * (1 - validation_size))
        return data.iloc[:split_idx], data.iloc[split_idx:]
    
    def _grid_search(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        param_grid: Dict[str, List[Any]],
        metric: str,
        cv_splits: int
    ) -> Dict[str, Any]:
        """Perform grid search optimization."""
        best_score = float("-inf")
        best_params = None
        results = []
        
        # Generate parameter combinations
        param_combinations = self._generate_param_combinations(param_grid)
        
        # Create cross-validation splits
        cv = TimeSeriesSplit(n_splits=cv_splits)
        
        # Evaluate each parameter combination
        for params in param_combinations:
            scores = []
            
            # Cross-validation
            for train_idx, val_idx in cv.split(train_data):
                cv_train = train_data.iloc[train_idx]
                cv_val = train_data.iloc[val_idx]
                
                # Create and evaluate strategy
                strategy = self.strategy_class(**params)
                score = self._evaluate_strategy(
                    strategy, cv_train, cv_val, metric
                )
                scores.append(score)
            
            # Calculate average score
            avg_score = np.mean(scores)
            
            # Store results
            results.append({
                "params": params,
                "score": avg_score,
                "std": np.std(scores)
            })
            
            # Update best parameters
            if avg_score > best_score:
                best_score = avg_score
                best_params = params
        
        # Final validation
        final_strategy = self.strategy_class(**best_params)
        final_score = self._evaluate_strategy(
            final_strategy, train_data, val_data, metric
        )
        
        return {
            "best_params": best_params,
            "best_score": best_score,
            "final_score": final_score,
            "results": results
        }
    
    def _random_search(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        param_grid: Dict[str, List[Any]],
        metric: str,
        n_trials: int,
        cv_splits: int
    ) -> Dict[str, Any]:
        """Perform random search optimization."""
        best_score = float("-inf")
        best_params = None
        results = []
        
        # Create cross-validation splits
        cv = TimeSeriesSplit(n_splits=cv_splits)
        
        # Random trials
        for _ in range(n_trials):
            # Generate random parameters
            params = self._generate_random_params(param_grid)
            scores = []
            
            # Cross-validation
            for train_idx, val_idx in cv.split(train_data):
                cv_train = train_data.iloc[train_idx]
                cv_val = train_data.iloc[val_idx]
                
                # Create and evaluate strategy
                strategy = self.strategy_class(**params)
                score = self._evaluate_strategy(
                    strategy, cv_train, cv_val, metric
                )
                scores.append(score)
            
            # Calculate average score
            avg_score = np.mean(scores)
            
            # Store results
            results.append({
                "params": params,
                "score": avg_score,
                "std": np.std(scores)
            })
            
            # Update best parameters
            if avg_score > best_score:
                best_score = avg_score
                best_params = params
        
        # Final validation
        final_strategy = self.strategy_class(**best_params)
        final_score = self._evaluate_strategy(
            final_strategy, train_data, val_data, metric
        )
        
        return {
            "best_params": best_params,
            "best_score": best_score,
            "final_score": final_score,
            "results": results
        }
    
    def _bayesian_optimization(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        param_grid: Dict[str, List[Any]],
        metric: str,
        n_trials: int,
        cv_splits: int
    ) -> Dict[str, Any]:
        """Perform Bayesian optimization using Optuna."""
        def objective(trial):
            # Generate parameters from trial
            params = {}
            for param_name, param_values in param_grid.items():
                if isinstance(param_values[0], int):
                    params[param_name] = trial.suggest_int(
                        param_name, min(param_values), max(param_values)
                    )
                elif isinstance(param_values[0], float):
                    params[param_name] = trial.suggest_float(
                        param_name, min(param_values), max(param_values)
                    )
                else:
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_values
                    )
            
            # Cross-validation
            cv = TimeSeriesSplit(n_splits=cv_splits)
            scores = []
            
            for train_idx, val_idx in cv.split(train_data):
                cv_train = train_data.iloc[train_idx]
                cv_val = train_data.iloc[val_idx]
                
                strategy = self.strategy_class(**params)
                score = self._evaluate_strategy(
                    strategy, cv_train, cv_val, metric
                )
                scores.append(score)
            
            return np.mean(scores)
        
        # Create study
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=self.random_state),
            pruner=MedianPruner()
        )
        
        # Optimize
        study.optimize(objective, n_trials=n_trials)
        
        # Get best parameters
        best_params = study.best_params
        best_score = study.best_value
        
        # Final validation
        final_strategy = self.strategy_class(**best_params)
        final_score = self._evaluate_strategy(
            final_strategy, train_data, val_data, metric
        )
        
        return {
            "best_params": best_params,
            "best_score": best_score,
            "final_score": final_score,
            "study": study
        }
    
    def _genetic_optimization(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        param_grid: Dict[str, List[Any]],
        metric: str,
        n_trials: int,
        cv_splits: int
    ) -> Dict[str, Any]:
        """Perform genetic algorithm optimization."""
        from deap import base, creator, tools, algorithms
        
        # Create fitness and individual classes
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        # Initialize toolbox
        toolbox = base.Toolbox()
        
        # Register parameter generation
        for param_name, param_values in param_grid.items():
            toolbox.register(
                f"attr_{param_name}",
                np.random.choice,
                param_values
            )
        
        # Create individual and population
        toolbox.register(
            "individual",
            tools.initCycle,
            creator.Individual,
            [getattr(toolbox, f"attr_{param}") for param in param_grid.keys()],
            n=1
        )
        toolbox.register(
            "population",
            tools.initRepeat,
            list,
            toolbox.individual
        )
        
        # Define evaluation function
        def evaluate(individual):
            params = dict(zip(param_grid.keys(), individual))
            
            # Cross-validation
            cv = TimeSeriesSplit(n_splits=cv_splits)
            scores = []
            
            for train_idx, val_idx in cv.split(train_data):
                cv_train = train_data.iloc[train_idx]
                cv_val = train_data.iloc[val_idx]
                
                strategy = self.strategy_class(**params)
                score = self._evaluate_strategy(
                    strategy, cv_train, cv_val, metric
                )
                scores.append(score)
            
            return (np.mean(scores),)
        
        # Register genetic operators
        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Create initial population
        pop = toolbox.population(n=50)
        
        # Run genetic algorithm
        result, logbook = algorithms.eaSimple(
            pop,
            toolbox,
            cxpb=0.7,
            mutpb=0.2,
            ngen=n_trials,
            verbose=True
        )
        
        # Get best individual
        best_individual = tools.selBest(result, k=1)[0]
        best_params = dict(zip(param_grid.keys(), best_individual))
        best_score = best_individual.fitness.values[0]
        
        # Final validation
        final_strategy = self.strategy_class(**best_params)
        final_score = self._evaluate_strategy(
            final_strategy, train_data, val_data, metric
        )
        
        return {
            "best_params": best_params,
            "best_score": best_score,
            "final_score": final_score,
            "logbook": logbook
        }
    
    def _evaluate_strategy(
        self,
        strategy: BaseStrategy,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        metric: str
    ) -> float:
        """Evaluate strategy performance using specified metric."""
        # Generate signals
        train_signals = strategy.generate_signals(train_data)
        val_signals = strategy.generate_signals(val_data)
        
        # Calculate returns
        train_returns = self._calculate_returns(train_data, train_signals)
        val_returns = self._calculate_returns(val_data, val_signals)
        
        # Calculate metric
        if metric == "sharpe_ratio":
            return self._calculate_sharpe_ratio(val_returns)
        elif metric == "sortino_ratio":
            return self._calculate_sortino_ratio(val_returns)
        elif metric == "calmar_ratio":
            return self._calculate_calmar_ratio(val_returns)
        elif metric == "total_return":
            return self._calculate_total_return(val_returns)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def _calculate_returns(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame
    ) -> pd.Series:
        """Calculate strategy returns."""
        # Implement returns calculation based on signals
        # This is a placeholder - implement actual logic
        return pd.Series(0.0, index=data.index)
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        return np.sqrt(252) * returns.mean() / returns.std()
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio."""
        if len(returns) < 2:
            return 0.0
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return float("inf")
        return np.sqrt(252) * returns.mean() / downside_returns.std()
    
    def _calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """Calculate Calmar ratio."""
        if len(returns) < 2:
            return 0.0
        total_return = (1 + returns).prod() - 1
        max_drawdown = self._calculate_max_drawdown(returns)
        if max_drawdown == 0:
            return float("inf")
        return total_return / max_drawdown
    
    def _calculate_total_return(self, returns: pd.Series) -> float:
        """Calculate total return."""
        return (1 + returns).prod() - 1
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns / running_max) - 1
        return abs(drawdown.min())
    
    def _generate_param_combinations(
        self,
        param_grid: Dict[str, List[Any]]
    ) -> List[Dict[str, Any]]:
        """Generate all parameter combinations from grid."""
        import itertools
        
        keys = param_grid.keys()
        values = param_grid.values()
        
        combinations = []
        for combination in itertools.product(*values):
            combinations.append(dict(zip(keys, combination)))
        
        return combinations
    
    def _generate_random_params(
        self,
        param_grid: Dict[str, List[Any]]
    ) -> Dict[str, Any]:
        """Generate random parameters from grid."""
        params = {}
        for param_name, param_values in param_grid.items():
            params[param_name] = np.random.choice(param_values)
        return params
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get optimization history."""
        return self.optimization_history
    
    def plot_optimization_results(
        self,
        results: Dict[str, Any],
        metric: str = "score"
    ) -> None:
        """Plot optimization results."""
        import matplotlib.pyplot as plt
        
        if "results" in results:
            # Plot parameter importance
            plt.figure(figsize=(12, 6))
            
            if "study" in results:
                # Optuna study
                optuna.visualization.plot_param_importances(results["study"])
            else:
                # Grid/Random search results
                param_scores = []
                for result in results["results"]:
                    param_scores.append({
                        "params": result["params"],
                        "score": result["score"]
                    })
                
                param_scores = pd.DataFrame(param_scores)
                param_scores.plot(kind="bar", x="params", y="score")
            
            plt.title("Parameter Importance")
            plt.tight_layout()
            plt.show()
            
            # Plot optimization progress
            plt.figure(figsize=(12, 6))
            
            if "study" in results:
                # Optuna study
                optuna.visualization.plot_optimization_history(results["study"])
            else:
                # Grid/Random search results
                scores = [r["score"] for r in results["results"]]
                plt.plot(scores)
                plt.title("Optimization Progress")
                plt.xlabel("Trial")
                plt.ylabel("Score")
            
            plt.tight_layout()
            plt.show() 