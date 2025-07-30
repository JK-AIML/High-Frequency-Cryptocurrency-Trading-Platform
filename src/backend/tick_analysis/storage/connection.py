"""
Database connection management for the Tick Data Analysis system.

This module provides utilities for managing database connections and connection pools.
"""

import logging
from contextlib import contextmanager
from typing import Generator, Optional

import psycopg2
from psycopg2 import pool, sql
from psycopg2.extensions import connection as PgConnection

from ..config.db import get_db_config

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages database connections and connection pooling."""
    
    _instance = None
    _pool = None
    
    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._config = get_db_config()
        return cls._instance
    
    def __init__(self):
        """Initialize the connection pool."""
        if self._pool is None:
            self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize the connection pool."""
        try:
            self._pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=self._config.DB_POOL_MIN,
                maxconn=self._config.DB_POOL_MAX,
                **self._config.get_connection_params()
            )
            logger.info("Initialized database connection pool")
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            raise
    
    @contextmanager
    def get_connection(self) -> Generator[PgConnection, None, None]:
        """Get a connection from the pool.
        
        Yields:
            A database connection from the pool.
            
        Raises:
            psycopg2.Error: If there's an error getting a connection.
        """
        conn = None
        try:
            conn = self._pool.getconn()
            conn.autocommit = False
            yield conn
            conn.commit()
        except psycopg2.Error as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                self._pool.putconn(conn)
    
    @contextmanager
    def get_cursor(self, **kwargs) -> Generator[PgConnection.cursor, None, None]:
        """Get a cursor from a connection in the pool.
        
        Args:
            **kwargs: Additional arguments to pass to cursor().
            
        Yields:
            A database cursor.
        """
        with self.get_connection() as conn:
            with conn.cursor(**kwargs) as cursor:
                try:
                    yield cursor
                except Exception as e:
                    logger.error(f"Cursor error: {e}")
                    raise
    
    def close_all(self):
        """Close all connections in the pool."""
        if self._pool:
            self._pool.closeall()
            logger.info("Closed all database connections")
            self._pool = None


def get_connection_manager() -> ConnectionManager:
    """Get the connection manager instance."""
    return ConnectionManager()


@contextmanager
def get_db_connection() -> Generator[PgConnection, None, None]:
    """Get a database connection from the pool.
    
    Example:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                result = cur.fetchone()
    """
    manager = get_connection_manager()
    with manager.get_connection() as conn:
        yield conn


@contextmanager
def get_db_cursor(**kwargs) -> Generator[PgConnection.cursor, None, None]:
    """Get a database cursor from the connection pool.
    
    Example:
        with get_db_cursor() as cur:
            cur.execute("SELECT 1")
            result = cur.fetchone()
    """
    with get_db_connection() as conn:
        with conn.cursor(**kwargs) as cursor:
            yield cursor


def execute_query(query: str, params: Optional[tuple] = None):
    """Execute a query and return the results.
    
    Args:
        query: The SQL query to execute.
        params: Parameters for the query.
        
    Returns:
        The query results as a list of tuples.
    """
    with get_db_cursor() as cursor:
        cursor.execute(query, params or ())
        return cursor.fetchall()


def execute_many(query: str, params_list: list):
    """Execute a query multiple times with different parameters.
    
    Args:
        query: The SQL query to execute.
        params_list: List of parameter tuples.
    """
    with get_db_cursor() as cursor:
        cursor.executemany(query, params_list)


def execute_script(script_path: str):
    """Execute a SQL script from a file.
    
    Args:
        script_path: Path to the SQL script file.
    """
    with open(script_path, 'r') as f:
        sql_script = f.read()
    
    with get_db_connection() as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            cursor.execute(sql_script)
