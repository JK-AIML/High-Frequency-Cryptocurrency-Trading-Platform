"""
Database utility functions.
"""
from typing import Any, Dict, List, Optional, Type, TypeVar, Generic, Union
from datetime import datetime, timedelta
import json

from sqlalchemy.orm import Session
from sqlalchemy import inspect, text, and_, or_
from sqlalchemy.exc import SQLAlchemyError

from .base import Base

ModelType = TypeVar("ModelType", bound=Base)

async def get_or_create(
    db: Session,
    model: Type[ModelType],
    defaults: Optional[Dict[str, Any]] = None,
    **kwargs
) -> tuple[ModelType, bool]:
    """
    Get an instance of the model if it exists, otherwise create it.
    
    Args:
        db: Database session
        model: SQLAlchemy model class
        defaults: Default values for creation
        **kwargs: Attributes to query by
        
    Returns:
        Tuple of (instance, created) where created is a boolean indicating
        if the instance was created
    """
    instance = db.query(model).filter_by(**kwargs).first()
    if instance:
        return instance, False
    
    params = {**kwargs, **(defaults or {})}
    instance = model(**params)
    
    try:
        db.add(instance)
        db.commit()
        db.refresh(instance)
        return instance, True
    except SQLAlchemyError:
        db.rollback()
        raise

def bulk_upsert(
    db: Session,
    model: Type[ModelType],
    data: List[Dict[str, Any]],
    update_on_conflict: List[str] = None,
    index_elements: List[str] = None
) -> int:
    """
    Perform a bulk upsert operation.
    
    Args:
        db: Database session
        model: SQLAlchemy model class
        data: List of dictionaries with data to upsert
        update_on_conflict: List of column names to update on conflict
        index_elements: List of column names that form the unique constraint
        
    Returns:
        Number of rows affected
    """
    if not data:
        return 0
    
    table = model.__table__
    
    # Default to primary key if no index elements provided
    if index_elements is None:
        index_elements = [col.name for col in table.primary_key.columns]
    
    # Default to all columns except primary key if no update columns provided
    if update_on_conflict is None:
        update_on_conflict = [
            col.name for col in table.columns 
            if col.name not in index_elements and not col.primary_key
        ]
    
    # Build the ON CONFLICT DO UPDATE clause
    stmt = table.insert().values(data)
    
    if update_on_conflict and index_elements:
        update_dict = {col: getattr(stmt.excluded, col) for col in update_on_conflict}
        stmt = stmt.on_conflict_do_update(
            index_elements=index_elements,
            set_=update_dict
        )
    
    try:
        result = db.execute(stmt)
        db.commit()
        return result.rowcount
    except SQLAlchemyError:
        db.rollback()
        raise

def delete_old_records(
    db: Session,
    model: Type[ModelType],
    timestamp_column: str,
    older_than_days: int
) -> int:
    """
    Delete records older than a specified number of days.
    
    Args:
        db: Database session
        model: SQLAlchemy model class
        timestamp_column: Name of the timestamp column
        older_than_days: Number of days to keep
        
    Returns:
        Number of rows deleted
    """
    cutoff_date = datetime.utcnow() - timedelta(days=older_than_days)
    
    try:
        result = db.query(model).filter(
            getattr(model, timestamp_column) < cutoff_date
        ).delete(synchronize_session=False)
        db.commit()
        return result
    except SQLAlchemyError:
        db.rollback()
        raise

def get_table_columns(model: Type[ModelType]) -> List[str]:
    """Get all column names for a model."""
    return [column.name for column in model.__table__.columns]

def model_to_dict(instance: ModelType) -> Dict[str, Any]:
    """Convert a SQLAlchemy model instance to a dictionary."""
    return {
        c.key: getattr(instance, c.key)
        for c in inspect(instance).mapper.column_attrs
    }

def execute_raw_sql(db: Session, sql: str, params: Optional[Dict] = None) -> List[Dict]:
    """
    Execute raw SQL and return results as a list of dictionaries.
    
    Args:
        db: Database session
        sql: SQL query string
        params: Dictionary of parameters for the query
        
    Returns:
        List of dictionaries representing the query results
    """
    try:
        result = db.execute(text(sql), params or {})
        return [dict(row._mapping) for row in result]
    except SQLAlchemyError:
        db.rollback()
        raise
