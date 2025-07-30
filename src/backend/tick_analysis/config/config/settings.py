"""
Application configuration settings with environment variable support.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from pydantic import Field, validator, PostgresDsn, HttpUrl, RedisDsn
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv
from functools import lru_cache

# Load environment variables from .env file
load_dotenv()

# Get logger
logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="",
        case_sensitive=True,
        validate_assignment=True,
        env_nested_delimiter="__",
    )

    # Application
    APP_NAME: str = "Crypto Trading System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "production"
    SECRET_KEY: str = Field(..., validation_alias="SECRET_KEY")
    API_PREFIX: str = "/api/v1"
    ALLOWED_HOSTS: List[str] = ["*"]
    CORS_ORIGINS: List[str] = ["*"]

    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    MODELS_DIR: Path = BASE_DIR / "models"
    LOGS_DIR: Path = BASE_DIR / "logs"
    CACHE_DIR: Path = BASE_DIR / ".cache"

    # API Keys
    CRYPTOCOMPARE_API_KEY: str = Field(..., validation_alias="CRYPTOCOMPARE_API_KEY")
    POLYGON_API_KEY: str = Field("", validation_alias="POLYGON_API_KEY")
    ALPHA_VANTAGE_API_KEY: str = Field("", validation_alias="ALPHA_VANTAGE_API_KEY")

    # Database
    DATABASE_URL: str = Field(..., validation_alias="DATABASE_URL")
    DATABASE_POOL_SIZE: int = 20
    DATABASE_POOL_MAX_OVERFLOW: int = 10
    DATABASE_ECHO: bool = False

    @validator("DATABASE_URL", pre=True)
    def validate_database_url(cls, v):
        """Validate and normalize database URL."""
        if isinstance(v, str) and v.startswith("sqlite"):
            return v
        try:
            return str(PostgresDsn(v))
        except Exception as e:
            if os.getenv("ENVIRONMENT") == "testing":
                return v
            raise ValueError(f"Invalid database URL: {e}")

    # Redis
    REDIS_URL: RedisDsn = Field(
        "redis://localhost:6379/0", validation_alias="REDIS_URL"
    )
    REDIS_PASSWORD: Optional[str] = Field(None, env="REDIS_PASSWORD")

    # Cache
    CACHE_TTL: int = 300  # 5 minutes
    CACHE_ENABLED: bool = True

    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT: str = "1000/day, 100/hour, 10/minute"

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = str(Path(BASE_DIR) / "logs" / "trading_system.log")
    LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
    LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

    # Log rotation
    LOG_MAX_BYTES: int = 10 * 1024 * 1024  # 10 MB
    LOG_BACKUP_COUNT: int = 5

    # Trading
    DEFAULT_EXCHANGE: str = "binance"
    DEFAULT_SYMBOL: str = "BTC/USDT"
    DEFAULT_TIMEFRAME: str = "1d"

    # Risk Management
    MAX_POSITION_SIZE: float = 0.1  # 10% of portfolio
    STOP_LOSS_PCT: float = 0.02  # 2%
    TAKE_PROFIT_PCT: float = 0.05  # 5%
    MAX_DAILY_LOSS: float = 0.05  # 5% of portfolio

    # Backtesting
    BACKTEST_INITIAL_CAPITAL: float = 100000.0
    BACKTEST_COMMISSION: float = 0.001  # 0.1%
    BACKTEST_SLIPPAGE: float = 0.0005  # 0.05%
    RISK_FREE_RATE: float = 0.0
    ANNUAL_TRADING_DAYS: int = 252

    # Feature Flags
    ENABLE_ML_STRATEGIES: bool = True
    ENABLE_REALTIME_TRADING: bool = False
    ENABLE_BACKTESTING: bool = True
    ENABLE_RISK_MANAGEMENT: bool = True

    # API Rate Limits (requests per minute)
    RATE_LIMITS: Dict[str, int] = {
        "cryptocompare": 3000,
        "polygon": 200,
        "alpha_vantage": 5,
        "binance": 1200,
        "coinbase": 100,
        "kraken": 60,
    }

    # Sentry (Error Tracking)
    SENTRY_DSN: Optional[HttpUrl] = None
    SENTRY_ENVIRONMENT: str = "production"

    # Monitoring
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9100

    @validator("ENVIRONMENT")
    def validate_environment(cls, v):
        """Validate environment is one of the allowed values."""
        allowed_environments = ["development", "staging", "production", "testing"]
        if v.lower() not in allowed_environments:
            raise ValueError(f"Environment must be one of {allowed_environments}")
        return v.lower()

    @validator("LOG_LEVEL")
    def validate_log_level(cls, v):
        """Validate log level is one of the allowed values."""
        allowed_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed_levels:
            raise ValueError(f"Log level must be one of {allowed_levels}")
        return v.upper()

    @validator("ALLOWED_HOSTS", "CORS_ORIGINS", pre=True)
    def parse_list_strings(cls, v):
        """Parse comma-separated string into list of strings."""
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    def create_dirs(self):
        """Create necessary directories."""
        for directory in [
            self.DATA_DIR,
            self.MODELS_DIR,
            self.LOGS_DIR,
            self.CACHE_DIR,
        ]:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")


# Create settings instance
@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    settings = Settings()
    settings.create_dirs()
    return settings


# Initialize settings
settings = get_settings()

# Configure logging
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": settings.LOG_FORMAT,
            "datefmt": settings.LOG_DATE_FORMAT,
        },
    },
    "handlers": {
        "console": {
            "level": "DEBUG",
            "class": "logging.StreamHandler",
            "formatter": "standard",
        },
        "file": {
            "level": "DEBUG",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": settings.LOG_FILE,
            "maxBytes": settings.LOG_MAX_BYTES,
            "backupCount": settings.LOG_BACKUP_COUNT,
            "formatter": "standard",
        },
    },
    "loggers": {
        "": {
            "handlers": ["console", "file"],
            "level": settings.LOG_LEVEL,
            "propagate": True,
        }
    },
}

try:
    import logging.config
    logging.config.dictConfig(LOGGING_CONFIG)
    # Create logs directory if it doesn't exist
    os.makedirs(Path(settings.LOG_FILE).parent, exist_ok=True)
except Exception as e:
    # If logging configuration fails, set up a basic console logger
    logging.basicConfig(level=logging.INFO)
    logging.warning(f"Failed to configure logging: {e}. Using basic console logging.")

# Log configuration at startup
if settings.DEBUG:
    logger.setLevel(logging.DEBUG)
    logger.debug("Debug mode is enabled")

logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
logger.info(f"Environment: {settings.ENVIRONMENT}")

# Export commonly used settings
API_CONFIG = {
    "CRYPTOCOMPARE_API_KEY": settings.CRYPTOCOMPARE_API_KEY,
    "POLYGON_API_KEY": settings.POLYGON_API_KEY,
    "ALPHA_VANTAGE_API_KEY": settings.ALPHA_VANTAGE_API_KEY,
    "RATE_LIMITS": settings.RATE_LIMITS,
}

TRADING_CONFIG = {
    "DEFAULT_EXCHANGE": settings.DEFAULT_EXCHANGE,
    "DEFAULT_SYMBOL": settings.DEFAULT_SYMBOL,
    "DEFAULT_TIMEFRAME": settings.DEFAULT_TIMEFRAME,
    "MAX_POSITION_SIZE": settings.MAX_POSITION_SIZE,
    "STOP_LOSS_PCT": settings.STOP_LOSS_PCT,
    "TAKE_PROFIT_PCT": settings.TAKE_PROFIT_PCT,
    "MAX_DAILY_LOSS": settings.MAX_DAILY_LOSS,
}

BACKTEST_CONFIG = {
    "INITIAL_CAPITAL": settings.BACKTEST_INITIAL_CAPITAL,
    "COMMISSION": settings.BACKTEST_COMMISSION,
    "SLIPPAGE": settings.BACKTEST_SLIPPAGE,
    "RISK_FREE_RATE": settings.RISK_FREE_RATE,
    "ANNUAL_TRADING_DAYS": settings.ANNUAL_TRADING_DAYS,
}

FEATURE_FLAGS = {
    "ENABLE_ML_STRATEGIES": settings.ENABLE_ML_STRATEGIES,
    "ENABLE_REALTIME_TRADING": settings.ENABLE_REALTIME_TRADING,
    "ENABLE_BACKTESTING": settings.ENABLE_BACKTESTING,
    "ENABLE_RISK_MANAGEMENT": settings.ENABLE_RISK_MANAGEMENT,
}
