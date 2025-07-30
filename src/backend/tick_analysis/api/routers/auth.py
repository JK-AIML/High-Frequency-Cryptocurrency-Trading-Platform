"""
Authentication and user management endpoints.
"""
from datetime import timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status, Body
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from ... import crud, models, schemas
from ...core import security
from ...core.config import settings
from ...db.session import get_db
from ..deps import get_current_active_user, get_current_user

router = APIRouter(prefix="/auth", tags=["authentication"])

@router.post("/login", response_model=schemas.Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
) -> Any:
    """
    OAuth2 compatible token login, get an access token for future requests
    """
    user = crud.user.authenticate(
        db, email=form_data.username, password=form_data.password
    )
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect email or password"
        )
    elif not crud.user.is_active(user):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Inactive user"
        )
    
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    return {
        "access_token": security.create_access_token(
            user.id, expires_delta=access_token_expires
        ),
        "token_type": "bearer",
    }

@router.post("/register", response_model=schemas.User)
def register(
    user_in: schemas.UserCreate, 
    db: Session = Depends(get_db)
) -> Any:
    """
    Create new user.
    """
    user = crud.user.get_by_email(db, email=user_in.email)
    if user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The user with this email already exists in the system.",
        )
    user = crud.user.create(db, obj_in=user_in)
    return user

@router.post("/reset-password", status_code=status.HTTP_200_OK)
def reset_password(
    email: str = Body(..., embed=True),
    db: Session = Depends(get_db)
) -> Any:
    """
    Request a password reset (stub).
    """
    # TODO: Implement real password reset logic (send email, etc.)
    user = crud.user.get_by_email(db, email=email)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return {"msg": f"Password reset link sent to {email} (stub)"}

@router.post("/verify-email", status_code=status.HTTP_200_OK)
def verify_email(
    email: str = Body(..., embed=True),
    code: str = Body(..., embed=True),
    db: Session = Depends(get_db)
) -> Any:
    """
    Verify email address (stub).
    """
    # TODO: Implement real email verification logic
    user = crud.user.get_by_email(db, email=email)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return {"msg": f"Email {email} verified (stub)"}

@router.get("/me", response_model=schemas.User)
def read_user_me(
    current_user: models.User = Depends(get_current_active_user),
) -> Any:
    """
    Get current user.
    """
    return current_user
