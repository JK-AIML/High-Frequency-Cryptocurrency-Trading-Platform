"""
Database migrations for the Tick Data Analysis system.

This package contains all database migration scripts and utilities
for managing database schema changes over time.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import importlib
import inspect
from datetime import datetime

from psycopg2 import sql
from psycopg2.extensions import cursor as PgCursor

from ...config.db import get_db_config
from ..connection import get_db_connection

logger = logging.getLogger(__name__)

# Directory where migration files are stored
MIGRATIONS_DIR = Path(__file__).parent / "versions"

# Table to track migrations
MIGRATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS _migrations (
    id SERIAL PRIMARY KEY,
    version INTEGER NOT NULL,
    name VARCHAR(255) NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(version, name)
);
"""


def ensure_migrations_table():
    """Ensure the migrations table exists."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(MIGRATIONS_TABLE)
            conn.commit()


def get_applied_migrations() -> List[Tuple[int, str]]:
    """Get a list of applied migrations.
    
    Returns:
        List of (version, name) tuples of applied migrations.
    """
    ensure_migrations_table()
    
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute(
                    "SELECT version, name FROM _migrations ORDER BY version ASC"
                )
                return cur.fetchall()
            except Exception as e:
                logger.error(f"Error getting applied migrations: {e}")
                return []


def get_pending_migrations() -> List[Tuple[int, str, str]]:
    """Get a list of pending migrations.
    
    Returns:
        List of (version, name, path) tuples of pending migrations.
    """
    applied = {name: version for version, name in get_applied_migrations()}
    pending = []
    
    # Ensure migrations directory exists
    MIGRATIONS_DIR.mkdir(exist_ok=True, parents=True)
    
    # Find all migration files
    for file in sorted(MIGRATIONS_DIR.glob("*.py")):
        if file.name == "__init__.py":
            continue
            
        # Parse version and name from filename: VERSION__NAME.py
        try:
            version_str, name = file.stem.split("__", 1)
            version = int(version_str.lstrip("V"))
        except (ValueError, AttributeError):
            logger.warning(f"Invalid migration filename: {file.name}")
            continue
        
        if name not in applied or applied[name] < version:
            pending.append((version, name, str(file)))
    
    return sorted(pending, key=lambda x: x[0])


def run_migration(version: int, name: str, path: str):
    """Run a single migration.
    
    Args:
        version: Migration version number.
        name: Migration name.
        path: Path to the migration file.
    """
    logger.info(f"Running migration {version}__{name}")
    
    # Import the migration module
    module_name = f"tick_analysis.storage.migrations.versions.V{version}__{name}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import migration {path}")
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Find the migration function
    migration_func = None
    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj) and hasattr(obj, "__migration__"):
            migration_func = obj
            break
    
    if not migration_func:
        raise ValueError(f"No migration function found in {path}")
    
    # Run the migration in a transaction
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            try:
                # Run the migration
                migration_func(cur)
                
                # Record the migration
                cur.execute(
                    """
                    INSERT INTO _migrations (version, name, applied_at)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (version, name) 
                    DO UPDATE SET applied_at = EXCLUDED.applied_at
                    """,
                    (version, name, datetime.utcnow())
                )
                
                conn.commit()
                logger.info(f"Successfully applied migration {version}__{name}")
                
            except Exception as e:
                conn.rollback()
                logger.error(f"Migration {version}__{name} failed: {e}")
                raise


def migrate():
    """Run all pending migrations."""
    logger.info("Checking for pending migrations...")
    
    pending = get_pending_migrations()
    if not pending:
        logger.info("No pending migrations.")
        return
    
    logger.info(f"Found {len(pending)} pending migrations.")
    
    for version, name, path in pending:
        try:
            run_migration(version, name, path)
        except Exception as e:
            logger.error(f"Failed to run migration {version}__{name}: {e}")
            raise


def create_migration(name: str):
    """Create a new migration file.
    
    Args:
        name: Name of the migration (will be converted to snake_case).
    """
    # Ensure migrations directory exists
    MIGRATIONS_DIR.mkdir(exist_ok=True, parents=True)
    
    # Get the next version number
    version = 1
    if MIGRATIONS_DIR.exists():
        for file in MIGRATIONS_DIR.glob("V*.py"):
            try:
                file_version = int(file.stem.split("__")[0][1:])
                version = max(version, file_version + 1)
            except (ValueError, IndexError):
                continue
    
    # Create migration filename
    safe_name = "".join(c if c.isalnum() else "_" for c in name).lower().strip("_")
    filename = f"V{version:04d}__{safe_name}.py"
    filepath = MIGRATIONS_DIR / filename
    
    # Create migration template
    template = f""""
"""Migration {version}: {name}"""

def upgrade(cursor):
    """Run the migration."""
    # Add your migration code here
    pass

# Mark this function as a migration
upgrade.__migration__ = True
"""
    
    # Write the migration file
    with open(filepath, "w") as f:
        f.write(template.strip() + "\n")
    
    logger.info(f"Created migration: {filepath}")
    return filepath
