"""
Database management commands for the Tick Data Analysis system.

This module provides command-line utilities for managing the database,
including migrations, schema management, and maintenance tasks.
"""

import click
import logging
from pathlib import Path
import sys
from typing import Optional

from tick_analysis.storage.migrations import migrate, create_migration, get_applied_migrations, get_pending_migrations
from tick_analysis.storage.connection import get_connection_manager
from tick_analysis.config.db import get_db_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@click.group()
def db():
    """Database management commands."""
    pass

@db.command()
@click.option('--dsn', help='Database connection string')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def init(dsn: Optional[str], verbose: bool):
    """Initialize the database."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Override DSN if provided
    if dsn:
        db_config = get_db_config()
        db_config.TIMESCALEDB_DSN = dsn
    
    logger.info("Initializing database...")
    
    try:
        # Test the connection
        with get_connection_manager().get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                logger.info("Database connection successful")
        
        # Run migrations
        migrate()
        logger.info("Database initialization complete")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}", exc_info=verbose)
        sys.exit(1)

@db.command()
@click.argument('name')
def new_migration(name: str):
    """Create a new migration file.
    
    Args:
        name: Name of the migration (will be converted to snake_case).
    """
    try:
        path = create_migration(name)
        click.echo(f"Created migration: {path}")
    except Exception as e:
        logger.error(f"Failed to create migration: {e}")
        sys.exit(1)

@db.command()
def migrate_db():
    """Run all pending migrations."""
    try:
        migrate()
        logger.info("Migrations complete")
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        sys.exit(1)

@db.command()
@click.option('--all', is_flag=True, help='Show all migrations, not just pending')
def status(all: bool):
    """Show migration status."""
    try:
        applied = get_applied_migrations()
        pending = get_pending_migrations()
        
        click.echo("=== Migration Status ===")
        click.echo(f"Applied: {len(applied)}")
        click.echo(f"Pending: {len(pending)}")
        
        if all:
            click.echo("\nApplied Migrations:")
            for version, name in applied:
                click.echo(f"  {version:04d} {name}")
        
        if pending:
            click.echo("\nPending Migrations:")
            for version, name, _ in pending:
                click.echo(f"  {version:04d} {name}")
        
    except Exception as e:
        logger.error(f"Failed to get migration status: {e}")
        sys.exit(1)

@db.command()
@click.option('--yes', is_flag=True, help='Skip confirmation prompt')
def reset(yes: bool):
    """Reset the database (drop all tables and data)."""
    if not yes:
        click.confirm('This will drop all tables and data. Are you sure?', abort=True)
    
    try:
        with get_connection_manager().get_connection() as conn:
            with conn.cursor() as cur:
                # Drop all tables, views, and other objects
                cur.execute("""
                    DROP SCHEMA public CASCADE;
                    CREATE SCHEMA public;
                    GRANT ALL ON SCHEMA public TO public;
                """)
                conn.commit()
        
        click.echo("Database reset complete")
    except Exception as e:
        logger.error(f"Failed to reset database: {e}")
        sys.exit(1)

@db.command()
@click.argument('query')
@click.option('--output', '-o', type=click.Path(), help='Output file (CSV)')
def query(query: str, output: Optional[str]):
    """Execute a raw SQL query."""
    try:
        with get_connection_manager().get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                
                # Get column names
                columns = [desc[0] for desc in cur.description] if cur.description else []
                
                # Fetch all rows
                rows = cur.fetchall()
                
                if output:
                    import csv
                    with open(output, 'w', newline='') as f:
                        writer = csv.writer(f)
                        if columns:
                            writer.writerow(columns)
                        writer.writerows(rows)
                    click.echo(f"Results written to {output}")
                else:
                    # Print results in a table
                    from tabulate import tabulate
                    if rows:
                        click.echo(tabulate(rows, headers=columns, tablefmt='psql'))
                    else:
                        click.echo("No results")
    except Exception as e:
        logger.error(f"Query failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    db()
