"""
Database base classes and utilities.
"""
from typing import Any, Dict, Optional, Type, TypeVar, Generic, List
from datetime import datetime

from sqlalchemy.orm import Session
from sqlalchemy.ext.declarative import as_declarative, declared_attr
from pydantic import BaseModel

# Type variables for generic base class
ModelType = TypeVar("ModelType")
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)

@as_declarative()
class Base:
    """Base class for all database models."""
    id: Any
    __name__: str
    
    # Generate __tablename__ automatically
    @declared_attr
    def __tablename__(cls) -> str:
        return cls.__name__.lower()

class CRUDBase(Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    """Base class for CRUD operations."""
    
    def __init__(self, model: Type[ModelType]):
        """Initialize with a model class."""
        self.model = model
    
    def get(self, db: Session, id: Any) -> Optional[ModelType]:
        """Get a single record by ID."""
        return db.query(self.model).filter(self.model.id == id).first()
    
    def get_multi(
        self, db: Session, *, skip: int = 0, limit: int = 100
    ) -> List[ModelType]:
        """Get multiple records with pagination."""
        return db.query(self.model).offset(skip).limit(limit).all()
    
    def create(self, db: Session, *, obj_in: CreateSchemaType) -> ModelType:
        """Create a new record."""
        obj_in_data = obj_in.dict()
        db_obj = self.model(**obj_in_data)
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj
    
    def update(
        self, 
        db: Session, 
        *, 
        db_obj: ModelType,
        obj_in: UpdateSchemaType
    ) -> ModelType:
        """Update a record."""
        update_data = obj_in.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(db_obj, field, value)
        
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj
    
    def remove(self, db: Session, *, id: int) -> Optional[ModelType]:
        """Remove a record by ID."""
        obj = db.query(self.model).get(id)
        if obj:
            db.delete(obj)
            db.commit()
        return obj

def get_db() -> Session:
    """
    Dependency function that yields database sessions.
    
    Yields:
        Session: Database session
    """
    from .session import async_session_factory
    
    db = async_session_factory()
    try:
        yield db
    finally:
        db.close()