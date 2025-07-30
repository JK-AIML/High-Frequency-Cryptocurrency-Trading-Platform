"""Configuration management for the Tick Data Analysis system."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml  # type: ignore
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


class Config:
    """Configuration manager that loads settings from YAML files and environment variables."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize configuration.

        Args:
            config_path: Path to the YAML configuration file.
        """
        self.config: Dict[str, Any] = {}
        if config_path and config_path.exists():
            self.load(config_path)

    def load(self, config_path: Path) -> None:
        """Load configuration from a YAML file."""
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f) or {}

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.

        Supports dot notation for nested keys (e.g., 'database.host').
        Environment variables take precedence over YAML config.
        """
        # First try environment variable
        env_key = key.upper().replace(".", "_")
        if env_key in os.environ:
            return os.environ[env_key]

        # Then try YAML config with dot notation
        value = self.config
        for part in key.split("."):
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        return value

    def __getitem__(self, key: str) -> Any:
        """Get a configuration value using dictionary syntax."""
        value = self.get(key)
        if value is None:
            raise KeyError(f"Configuration key '{key}' not found")
        return value

    def __contains__(self, key: str) -> bool:
        """Check if a configuration key exists."""
        try:
            self[key]
            return True
        except KeyError:
            return False


# Global configuration instance
config = Config()


def load_config(config_path: Path) -> None:
    """Load configuration from a YAML file into the global config."""
    global config
    config = Config(config_path)
