"""
Database package for the Tick Data Analysis & Alpha Detection application.

This package contains all database-related code including models, sessions,
and database utilities.
"""
from .base import Base, CRUDBase, ModelType, CreateSchemaType, UpdateSchemaType
from .session import (
    engine,
    async_session_factory,
    get_db,
    init_db,
    close_db
)
from .utils import (
    get_or_create,
    bulk_upsert,
    delete_old_records,
    get_table_columns,
    model_to_dict,
    execute_raw_sql
)

__all__ = [
    # Base classes
    'Base',
    'CRUDBase',
    'ModelType',
    'CreateSchemaType',
    'UpdateSchemaType',
    
    # Session management
    'engine',
    'async_session_factory',
    'get_db',
    'init_db',
    'close_db',
    
    # Utilities
    'get_or_create',
    'bulk_upsert',
    'delete_old_records',
    'get_table_columns',
    'model_to_dict',
    'execute_raw_sql',
]
