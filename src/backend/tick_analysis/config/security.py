import os
from pathlib import Path
from typing import Optional
import secrets
from cryptography.fernet import Fernet
import base64
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class SecurityConfig:
    def __init__(self):
        self._load_environment()
        self._setup_encryption()
        
    def _load_environment(self):
        """Load environment variables from .env file."""
        env_path = Path('.env')
        if not env_path.exists():
            logger.warning("No .env file found. Using environment variables.")
        load_dotenv(env_path)
        
    def _setup_encryption(self):
        """Setup encryption key for sensitive data."""
        # Get or generate encryption key
        encryption_key = os.getenv('ENCRYPTION_KEY')
        if not encryption_key:
            # Generate a new key if none exists
            encryption_key = Fernet.generate_key()
            logger.warning("No encryption key found. Generated new key. Please save this key securely!")
            print(f"Generated encryption key: {encryption_key.decode()}")
        
        # Initialize Fernet with the key
        self.fernet = Fernet(encryption_key if isinstance(encryption_key, bytes) else encryption_key.encode())
        
    def encrypt_value(self, value: str) -> str:
        """Encrypt a string value."""
        try:
            return self.fernet.encrypt(value.encode()).decode()
        except Exception as e:
            logger.error(f"Error encrypting value: {str(e)}")
            raise
            
    def decrypt_value(self, encrypted_value: str) -> str:
        """Decrypt an encrypted string value."""
        try:
            return self.fernet.decrypt(encrypted_value.encode()).decode()
        except Exception as e:
            logger.error(f"Error decrypting value: {str(e)}")
            raise
            
    def get_secure_value(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get a secure value from environment variables."""
        value = os.getenv(key, default)
        if value and key.startswith('SECURE_'):
            try:
                return self.decrypt_value(value)
            except Exception as e:
                logger.error(f"Error decrypting secure value for {key}: {str(e)}")
                return default
        return value

# Create a singleton instance
security_config = SecurityConfig()

def get_security_config() -> SecurityConfig:
    """Get the security configuration singleton instance."""
    return security_config 