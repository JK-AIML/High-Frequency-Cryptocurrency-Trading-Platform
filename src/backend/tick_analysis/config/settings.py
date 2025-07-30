"""Settings for tick analysis config."""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    """Application settings."""
    
    # Debug mode
    debug: bool = Field(default=False, env="DEBUG")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: Optional[Path] = Field(default=None, env="LOG_FILE")
    
    # Test mode
    test_mode: bool = Field(default=False, env="TESTING")
    
    # Data directories
    data_dir: Path = Field(default=Path("./data"), env="DATA_DIR")
    cache_dir: Path = Field(default=Path("./data/cache"), env="CACHE_DIR")
    
    # Database
    database_url: str = Field(
        default="sqlite:///./data/tick_analysis.db",
        env="DATABASE_URL"
    )
    
    # Redis
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        env="REDIS_URL"
    )
    
    # API
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=4, env="API_WORKERS")
    
    # WebSocket
    ws_host: str = Field(default="0.0.0.0", env="WS_HOST")
    ws_port: int = Field(default=8001, env="WS_PORT")
    
    # Trading
    default_commission: float = Field(default=0.001, env="DEFAULT_COMMISSION")
    default_slippage: float = Field(default=0.0005, env="DEFAULT_SLIPPAGE")
    
    # Risk management
    max_position_size: float = Field(default=0.1, env="MAX_POSITION_SIZE")
    max_drawdown: float = Field(default=0.2, env="MAX_DRAWDOWN")
    
    # Backtesting
    backtest_start_date: Optional[str] = Field(default=None, env="BACKTEST_START_DATE")
    backtest_end_date: Optional[str] = Field(default=None, env="BACKTEST_END_DATE")
    
    # Monitoring
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    alert_email: Optional[str] = Field(default=None, env="ALERT_EMAIL")
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
        "extra": "allow"
    }
    
    def __init__(self, **kwargs):
        """Initialize settings."""
        super().__init__(**kwargs)
        
        # Create directories if they don't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
