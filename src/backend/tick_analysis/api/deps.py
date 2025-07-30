"""
Dependencies for FastAPI routes.
"""
from typing import Generator, Optional

from fastapi import Depends, HTTPException, status, Security
from fastapi.security import OAuth2PasswordBearer, SecurityScopes
from jose import JWTError, jwt
from pydantic import ValidationError
from sqlalchemy.orm import Session

from .. import models, schemas, crud
from ..core.config import settings
from ..db.session import SessionLocal
from ..cache.redis_manager import redis_manager

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V1_STR}/auth/login",
    scopes={
        "user": "Read information about the current user.",
        "tick:read": "Read tick data.",
        "tick:write": "Create or update tick data.",
        "admin": "Admin operations.",
    }
)

def get_db() -> Generator:
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def get_current_user(
    security_scopes: SecurityScopes,
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> models.User:
    """Get current active user from token with scope validation."""
    if security_scopes.scopes:
        authenticate_value = f'Bearer scope=\"{security_scopes.scope_str}\"'
    else:
        authenticate_value = "Bearer"
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": authenticate_value},
    )
    
    try:
        # Try to get from cache first
        cache_key = f"user_token:{token}"
        user_data = await redis_manager.get(cache_key)
        
        if user_data is None:
            # If not in cache, decode the token
            payload = jwt.decode(
                token, 
                settings.SECRET_KEY, 
                algorithms=[settings.JWT_ALGORITHM]
            )
            username: str = payload.get("sub")
            if username is None:
                raise credentials_exception
            
            # Get user from database
            user = crud.user.get_by_username(db, username=username)
            if user is None:
                raise credentials_exception
                
            # Cache the user data
            await redis_manager.set(cache_key, user.to_dict(), ttl=300)
        else:
            # Create user from cached data
            user = models.User(**user_data)
            
        # Check scopes
        if security_scopes.scopes and not user.is_superuser:
            token_scopes = payload.get("scopes", [])
            for scope in security_scopes.scopes:
                if scope not in token_scopes:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Not enough permissions",
                        headers={"WWW-Authenticate": authenticate_value},
                    )
                    
        return user
        
    except (JWTError, ValidationError) as e:
        logger.error(f"Authentication error: {str(e)}")
        raise credentials_exception

def get_current_active_user(
    current_user: models.User = Security(get_current_user, scopes=["user"])
) -> models.User:
    """Get current active user."""
    if not crud.user.is_active(current_user):
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

def get_current_active_superuser(
    current_user: models.User = Security(get_current_user, scopes=["admin"])
) -> models.User:
    """Get current active superuser."""
    if not crud.user.is_superuser(current_user):
        raise HTTPException(
            status_code=400, 
            detail="The user doesn't have enough privileges"
        )
    return current_user
