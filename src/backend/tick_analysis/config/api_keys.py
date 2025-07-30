from pydantic_settings import BaseSettings
from pydantic import SecretStr
from typing import Optional
import os
from dotenv import load_dotenv
from .security import get_security_config

# Load environment variables
load_dotenv()

class APIConfig(BaseSettings):
    # Polygon.io API
    POLYGON_API_KEY: SecretStr = SecretStr(os.getenv('POLYGON_API_KEY', ''))
    
    # Binance API
    BINANCE_API_KEY: SecretStr = SecretStr(os.getenv('BINANCE_API_KEY', ''))
    BINANCE_API_SECRET: SecretStr = SecretStr(os.getenv('BINANCE_API_SECRET', ''))
    
    # CryptoCompare API
    CRYPTOCOMPARE_API_KEY: SecretStr = SecretStr(os.getenv('CRYPTOCOMPARE_API_KEY', ''))
    
    # JWT Configuration
    JWT_SECRET_KEY: SecretStr = SecretStr(os.getenv('JWT_SECRET_KEY', ''))
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    class Config:
        env_file = ".env"
        case_sensitive = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._secure_config = get_security_config()
        self._decrypt_secrets()

    def _decrypt_secrets(self):
        """Decrypt any encrypted secrets."""
        try:
            # Decrypt API keys if they are encrypted
            if self.POLYGON_API_KEY.get_secret_value().startswith('SECURE_'):
                decrypted = self._secure_config.decrypt_value(self.POLYGON_API_KEY.get_secret_value())
                self.POLYGON_API_KEY = SecretStr(decrypted)

            if self.BINANCE_API_KEY.get_secret_value().startswith('SECURE_'):
                decrypted = self._secure_config.decrypt_value(self.BINANCE_API_KEY.get_secret_value())
                self.BINANCE_API_KEY = SecretStr(decrypted)

            if self.BINANCE_API_SECRET.get_secret_value().startswith('SECURE_'):
                decrypted = self._secure_config.decrypt_value(self.BINANCE_API_SECRET.get_secret_value())
                self.BINANCE_API_SECRET = SecretStr(decrypted)

            if self.CRYPTOCOMPARE_API_KEY.get_secret_value().startswith('SECURE_'):
                decrypted = self._secure_config.decrypt_value(self.CRYPTOCOMPARE_API_KEY.get_secret_value())
                self.CRYPTOCOMPARE_API_KEY = SecretStr(decrypted)

            if self.JWT_SECRET_KEY.get_secret_value().startswith('SECURE_'):
                decrypted = self._secure_config.decrypt_value(self.JWT_SECRET_KEY.get_secret_value())
                self.JWT_SECRET_KEY = SecretStr(decrypted)

        except Exception as e:
            raise ValueError(f"Error decrypting secrets: {str(e)}")

# Create a singleton instance
api_config = APIConfig()

def get_api_config() -> APIConfig:
    """Get the API configuration singleton instance."""
    return api_config 