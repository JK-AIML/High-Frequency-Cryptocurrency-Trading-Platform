"""
Database configuration for the Tick Data Analysis system.

This module provides configuration for connecting to TimescaleDB and other databases.
"""

import os
from typing import Dict, Optional
from pydantic import BaseSettings, PostgresDsn, validator


class DatabaseConfig(BaseSettings):
    """Database configuration settings."""
    
    # TimescaleDB configuration
    TIMESCALEDB_HOST: str = "localhost"
    TIMESCALEDB_PORT: int = 5432
    TIMESCALEDB_USER: str = "postgres"
    TIMESCALEDB_PASSWORD: str = "postgres"
    TIMESCALEDB_DB: str = "tickdata"
    TIMESCALEDB_SCHEMA: str = "public"
    TIMESCALEDB_SSLMODE: str = "prefer"
    
    # Connection pool settings
    DB_POOL_MIN: int = 1
    DB_POOL_MAX: int = 10
    DB_POOL_TIMEOUT: int = 30  # seconds
    DB_STATEMENT_TIMEOUT: int = 60  # seconds
    
    # Migration settings
    DB_RUN_MIGRATIONS: bool = True
    
    class Config:
        env_prefix = "DB_"
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    @property
    def timescaledb_dsn(self) -> str:
        """Generate a DSN string for TimescaleDB."""
        return (
            f"postgresql://{self.TIMESCALEDB_USER}:"
            f"{self.TIMESCALEDB_PASSWORD}@"
            f"{self.TIMESCALEDB_HOST}:"
            f"{self.TIMESCALEDB_PORT}/"
            f"{self.TIMESCALEDB_DB}?"
            f"sslmode={self.TIMESCALEDB_SSLMODE}"
        )
    
    @property
    def sqlalchemy_database_uri(self) -> str:
        """Generate a SQLAlchemy-compatible database URI."""
        return self.timescaledb_dsn
    
    def get_connection_params(self) -> Dict[str, str]:
        """Get connection parameters as a dictionary."""
        return {
            "host": self.TIMESCALEDB_HOST,
            "port": str(self.TIMESCALEDB_PORT),
            "user": self.TIMESCALEDB_USER,
            "password": self.TIMESCALEDB_PASSWORD,
            "dbname": self.TIMESCALEDB_DB,
            "sslmode": self.TIMESCALEDB_SSLMODE,
            "options": f"-c search_path={self.TIMESCALEDB_SCHEMA},public"
        }


# Create a singleton instance
db_config = DatabaseConfig()


def get_db_config() -> DatabaseConfig:
    """Get the database configuration."""
    return db_config
