"""
CRUD operations for users.
"""
from typing import Optional, List, Any, Dict, Union
from datetime import datetime

from sqlalchemy.orm import Session
from sqlalchemy import or_

from ..core.security import get_password_hash, verify_password
from ..db.models import User
from ..schemas.user import UserCreate, UserUpdate, UserInDB

def get_user(db: Session, user_id: int) -> Optional[User]:
    """Get a user by ID."""
    return db.query(User).filter(User.id == user_id).first()

def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Get a user by email."""
    return db.query(User).filter(User.email == email).first()

def get_user_by_username(db: Session, username: str) -> Optional[User]:
    """Get a user by username."""
    return db.query(User).filter(User.username == username).first()

def get_users(
    db: Session, 
    skip: int = 0, 
    limit: int = 100,
    search: Optional[str] = None
) -> List[User]:
    """Get multiple users with optional search."""
    query = db.query(User)
    
    if search:
        search = f"%{search}%"
        query = query.filter(
            or_(
                User.email.ilike(search),
                User.username.ilike(search),
                User.full_name.ilike(search)
            )
        )
    
    return query.offset(skip).limit(limit).all()

def create_user(db: Session, user_in: UserCreate) -> User:
    """Create a new user."""
    db_user = User(
        email=user_in.email,
        username=user_in.username,
        hashed_password=get_password_hash(user_in.password),
        full_name=user_in.full_name,
        is_active=user_in.is_active,
        is_superuser=user_in.is_superuser
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def update_user(
    db: Session, 
    db_user: User, 
    user_in: Union[UserUpdate, Dict[str, Any]]
) -> User:
    """Update a user."""
    user_data = user_in.dict(exclude_unset=True) if not isinstance(user_in, dict) else user_in
    
    if "password" in user_data:
        hashed_password = get_password_hash(user_data["password"])
        del user_data["password"]
        user_data["hashed_password"] = hashed_password
    
    for field, value in user_data.items():
        setattr(db_user, field, value)
    
    db_user.updated_at = datetime.utcnow()
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def delete_user(db: Session, user_id: int) -> User:
    """Delete a user."""
    db_user = db.query(User).filter(User.id == user_id).first()
    if db_user:
        db.delete(db_user)
        db.commit()
    return db_user

def authenticate(
    db: Session, 
    email: str, 
    password: str
) -> Optional[User]:
    """Authenticate a user."""
    user = get_user_by_email(db, email=email)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user

def is_active(user: User) -> bool:
    """Check if a user is active."""
    return user.is_active

def is_superuser(user: User) -> bool:
    """Check if a user is a superuser."""
    return user.is_superuser
