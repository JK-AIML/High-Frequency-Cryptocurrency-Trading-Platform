import os
import sys
import logging
from pathlib import Path

# Add the project root directory to the Python path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

from src.tick_analysis.database import init_db, drop_db
from src.tick_analysis.models.base import Base, User
from sqlalchemy.orm import Session
from src.tick_analysis.database import SessionLocal
from passlib.context import CryptContext

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def create_initial_user(db: Session):
    """Create an initial admin user if no users exist."""
    if db.query(User).first() is None:
        admin_user = User(
            username="admin",
            email="admin@example.com",
            hashed_password=pwd_context.hash("admin123"),
            full_name="Admin User",
            disabled=False
        )
        db.add(admin_user)
        db.commit()
        logger.info("Created initial admin user")
    else:
        logger.info("Users already exist, skipping initial user creation")

def main():
    """Initialize the database and create initial data."""
    try:
        # Drop existing tables
        logger.info("Dropping existing tables...")
        drop_db()
        
        # Create tables
        logger.info("Creating tables...")
        init_db()
        
        # Create initial user
        logger.info("Creating initial user...")
        db = SessionLocal()
        try:
            create_initial_user(db)
        finally:
            db.close()
        
        logger.info("Database initialization completed successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 