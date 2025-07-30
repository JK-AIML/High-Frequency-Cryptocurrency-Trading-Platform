"""
Configuration management for the Tick Data Analysis system.
"""
import os
from typing import Dict, Any, Optional
from pydantic import BaseSettings, PostgresDsn, RedisDsn, validator
from dotenv import load_dotenv
import logging

# Load environment variables from .env file if it exists
load_dotenv()

logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    """Application settings."""
    
    # Application
    APP_NAME: str = "Tick Data Analysis"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"
    SECRET_KEY: str = "your-secret-key-here"  # Change in production!
    
    # API
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "tick-analysis"
    
    # CORS
    BACKEND_CORS_ORIGINS: list[str] = ["*"]
    
    # Database
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "tickdb"
    DATABASE_URI: Optional[PostgresDsn] = None
    
    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    REDIS_URI: Optional[RedisDsn] = None
    
    # JWT
    JWT_SECRET_KEY: str = "your-jwt-secret"  # Change in production!
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Feature Flags
    ENABLE_CACHING: bool = True
    ENABLE_RATE_LIMITING: bool = True
    
    class Config:
        case_sensitive = True
        env_file = ".env"
        
    @validator("DATABASE_URI", pre=True)
    def assemble_db_connection(cls, v: Optional[str], values: Dict[str, Any]) -> Any:
        if isinstance(v, str):
            return v
        return PostgresDsn.build(
            scheme="postgresql",
            user=values.get("POSTGRES_USER"),
            password=values.get("POSTGRES_PASSWORD"),
            host=values.get("POSTGRES_SERVER"),
            path=f"/{values.get('POSTGRES_DB') or ''}",
        )
    
    @validator("REDIS_URI", pre=True)
    def assemble_redis_connection(cls, v: Optional[str], values: Dict[str, Any]) -> Any:
        if isinstance(v, str):
            return v
        return RedisDsn.build(
            scheme="redis",
            host=values.get("REDIS_HOST"),
            port=str(values.get("REDIS_PORT")),
            path=f"/{values.get('REDIS_DB', 0)}",
            user=None,
            password=values.get("REDIS_PASSWORD") or None,
        )

# Create settings instance
settings = Settings()

# Update logging configuration
def setup_logging():
    """Configure logging based on settings."""
    import logging.config
    import logging.handlers
    import sys
    
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": settings.LOG_FORMAT,
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "json": {
                "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                "format": """
                    asctime: %(asctime)s
                    created: %(created)f
                    filename: %(filename)s
                    funcName: %(funcName)s
                    levelname: %(levelname)s
                    levelno: %(levelno)s
                    lineno: %(lineno)d
                    message: %(message)s
                    module: %(module)s
                    msecs: %(msecs)d
                    name: %(name)s
                    pathname: %(pathname)s
                    process: %(process)d
                    processName: %(processName)s
                    relativeCreated: %(relativeCreated)d
                    thread: %(thread)d
                    threadName: %(threadName)s
                """,
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "level": log_level,
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "stream": sys.stdout,
            },
            "file": {
                "level": log_level,
                "class": "logging.handlers.RotatingFileHandler",
                "filename": "tick_analysis.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "formatter": "standard",
                "encoding": "utf8",
            },
        },
        "loggers": {
            "": {
                "handlers": ["console", "file"],
                "level": log_level,
                "propagate": True,
            },
            "uvicorn": {
                "handlers": ["console", "file"],
                "level": log_level,
                "propagate": False,
            },
            "uvicorn.error": {
                "level": log_level,
                "handlers": ["console", "file"],
                "propagate": False,
            },
            "uvicorn.access": {
                "handlers": ["console", "file"],
                "level": log_level,
                "propagate": False,
            },
        },
    }
    
    logging.config.dictConfig(logging_config)
    logger.info("Logging configured")

# Initialize logging when module is imported
setup_logging()
