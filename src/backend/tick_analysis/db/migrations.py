"""
Database migration utilities for the Tick Data Analysis & Alpha Detection application.

This module provides functions to manage database migrations, including creating
and applying migrations, as well as rolling back changes when needed.
"""
import os
import logging
from datetime import datetime
from typing import List, Optional, Tuple
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from . import Base, engine, init_db
from .models import User, Portfolio, Position, TickData

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
ALEMBIC_INI = PROJECT_ROOT / "alembic.ini"
ALEMBIC_DIR = PROJECT_ROOT / "alembic"

def ensure_alembic_config() -> Config:
    """Ensure the Alembic configuration exists and return the config object."""
    if not ALEMBIC_INI.exists():
        raise FileNotFoundError(
            f"Alembic config file not found at {ALEMBIC_INI}. "
            "Please run 'alembic init alembic' in the project root."
        )
    
    # Create the alembic directory if it doesn't exist
    ALEMBIC_DIR.mkdir(exist_ok=True)
    
    # Configure Alembic
    config = Config(ALEMBIC_INI)
    config.set_main_option('script_location', str(ALEMBIC_DIR))
    config.set_main_option('sqlalchemy.url', str(engine.url))
    
    return config

def create_migration(message: str = None) -> Optional[str]:
    """
    Create a new database migration.
    
    Args:
        message: Description of the migration
        
    Returns:
        Path to the new migration file, or None if no changes were detected
    """
    try:
        config = ensure_alembic_config()
        
        # Generate a timestamp for the migration
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        # Create a descriptive name for the migration
        name = f"{timestamp}_migration"
        if message:
            # Convert message to a filesystem-safe string
            safe_message = "".join(
                c if c.isalnum() or c in "_-" else "_" 
                for c in message.lower().replace(" ", "_")
            )
            name = f"{timestamp}_{safe_message}"
        
        # Create the migration
        migration_path = command.revision(
            config=config,
            autogenerate=True,
            message=message or "auto-generated migration",
            rev_id=timestamp,
            version_path=str(ALEMBIC_DIR / "versions"),
        )
        
        logger.info(f"Created migration: {migration_path}")
        return migration_path
    except Exception as e:
        logger.error(f"Failed to create migration: {e}")
        raise

def upgrade_db(revision: str = "head") -> None:
    """
    Upgrade the database to the specified revision.
    
    Args:
        revision: Target revision (defaults to 'head')
    """
    try:
        config = ensure_alembic_config()
        command.upgrade(config, revision)
        logger.info(f"Database upgraded to revision: {revision}")
    except Exception as e:
        logger.error(f"Failed to upgrade database: {e}")
        raise

def downgrade_db(revision: str) -> None:
    """
    Downgrade the database to the specified revision.
    
    Args:
        revision: Target revision
    """
    try:
        config = ensure_alembic_config()
        command.downgrade(config, revision)
        logger.info(f"Database downgraded to revision: {revision}")
    except Exception as e:
        logger.error(f"Failed to downgrade database: {e}")
        raise

def reset_db() -> None:
    """
    Reset the database by dropping all tables and recreating them.
    
    WARNING: This will delete all data in the database!
    """
    logger.warning("Dropping all database tables...")
    
    try:
        # Drop all tables
        Base.metadata.drop_all(bind=engine)
        
        # Recreate all tables
        Base.metadata.create_all(bind=engine)
        
        logger.info("Database reset complete")
    except SQLAlchemyError as e:
        logger.error(f"Failed to reset database: {e}")
        raise

def check_migrations() -> Tuple[bool, List[str]]:
    """
    Check if there are any pending migrations.
    
    Returns:
        Tuple of (is_up_to_date, pending_migrations)
    """
    try:
        config = ensure_alembic_config()
        
        # Get the current database revision
        current_rev = command.current(config)
        
        # Get the latest revision
        heads = command.heads(config)
        
        # Check if we're up to date
        is_up_to_date = current_rev in heads
        
        # Get the list of pending migrations
        pending = []
        if not is_up_to_date:
            # Get the list of revisions between current and head
            history = command.history(config)
            pending = [rev.revision for rev in history if rev.revision != current_rev]
        
        return is_up_to_date, pending
    except Exception as e:
        logger.error(f"Failed to check migrations: {e}")
        raise
