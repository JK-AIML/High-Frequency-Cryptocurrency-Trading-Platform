"""
Central exceptions module for the tick analysis system.
"""

# Base exceptions
class TickAnalysisError(Exception):
    """Base exception for all tick analysis errors."""
    pass

class ConfigurationError(TickAnalysisError):
    """Raised when there's a configuration error."""
    pass

class DataError(TickAnalysisError):
    """Base exception for data-related errors."""
    pass

class ValidationError(TickAnalysisError):
    """Raised when data validation fails."""
    pass

# Stream processing exceptions
class StreamProcessingError(TickAnalysisError):
    """Raised when stream processing fails."""
    pass

class StreamConnectionError(StreamProcessingError):
    """Raised when stream connection fails."""
    pass

class StreamTimeoutError(StreamProcessingError):
    """Raised when stream operation times out."""
    pass

# Database exceptions
class DatabaseError(TickAnalysisError):
    """Base exception for database errors."""
    pass

class ConnectionError(DatabaseError):
    """Raised when database connection fails."""
    pass

class MigrationError(DatabaseError):
    """Raised when database migration fails."""
    pass

# Execution exceptions
class ExecutionError(TickAnalysisError):
    """Base exception for execution errors."""
    pass

class OrderRejected(ExecutionError):
    """Raised when an order is rejected by the exchange or broker."""
    pass

class InsufficientFunds(ExecutionError):
    """Raised when there are not enough funds to execute an order."""
    pass

class InvalidOrder(ExecutionError):
    """Raised when an order is invalid."""
    pass

class MarketClosed(ExecutionError):
    """Raised when trying to execute an order while the market is closed."""
    pass

class PositionLimitExceeded(ExecutionError):
    """Raised when a position limit is exceeded."""
    pass

# Strategy exceptions
class StrategyError(TickAnalysisError):
    """Base exception for strategy errors."""
    pass

class StrategyConfigurationError(StrategyError):
    """Raised when strategy configuration is invalid."""
    pass

class StrategyExecutionError(StrategyError):
    """Raised when strategy execution fails."""
    pass

# Portfolio exceptions
class PortfolioError(TickAnalysisError):
    """Base exception for portfolio errors."""
    pass

class InsufficientCapital(PortfolioError):
    """Raised when there's insufficient capital for a trade."""
    pass

class RiskLimitExceeded(PortfolioError):
    """Raised when a risk limit is exceeded."""
    pass

# Risk management exceptions
class RiskManagementError(TickAnalysisError):
    """Base exception for risk management errors."""
    pass

class VaRLimitExceeded(RiskManagementError):
    """Raised when VaR limit is exceeded."""
    pass

class DrawdownLimitExceeded(RiskManagementError):
    """Raised when drawdown limit is exceeded."""
    pass

# API exceptions
class APIError(TickAnalysisError):
    """Base exception for API errors."""
    pass

class AuthenticationError(APIError):
    """Raised when API authentication fails."""
    pass

class RateLimitExceeded(APIError):
    """Raised when API rate limit is exceeded."""
    pass

class APIConnectionError(APIError):
    """Raised when API connection fails."""
    pass

# Monitoring exceptions
class MonitoringError(TickAnalysisError):
    """Base exception for monitoring errors."""
    pass

class AlertError(MonitoringError):
    """Raised when alert processing fails."""
    pass

class MetricError(MonitoringError):
    """Raised when metric calculation fails."""
    pass

# Data pipeline exceptions
class PipelineError(TickAnalysisError):
    """Base exception for data pipeline errors."""
    pass

class DataTransformationError(PipelineError):
    """Raised when data transformation fails."""
    pass

class DataValidationError(PipelineError):
    """Raised when data validation fails."""
    pass

# Backtesting exceptions
class BacktestError(TickAnalysisError):
    """Base exception for backtesting errors."""
    pass

class InsufficientDataError(BacktestError):
    """Raised when there's insufficient data for backtesting."""
    pass

class BacktestConfigurationError(BacktestError):
    """Raised when backtest configuration is invalid."""
    pass 