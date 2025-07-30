"""
User management endpoints.
"""
from typing import Any, List

from fastapi import APIRouter, Depends, HTTPException, status, Body
from sqlalchemy.orm import Session

from ... import crud, models, schemas
from ...core.config import settings
from ...db.session import get_db
from ..deps import get_current_active_superuser, get_current_user

router = APIRouter(prefix="/users", tags=["users"])

@router.get("/", response_model=List[schemas.User])
def read_users(
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: models.User = Depends(get_current_active_superuser),
) -> Any:
    """
    Retrieve users.
    """
    users = crud.user.get_multi(db, skip=skip, limit=limit)
    return users

@router.post("/", response_model=schemas.User)
def create_user(
    *,
    db: Session = Depends(get_db),
    user_in: schemas.UserCreate,
    current_user: models.User = Depends(get_current_active_superuser),
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

@router.get("/{user_id}", response_model=schemas.User)
def read_user(
    user_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Any:
    """
    Get a specific user by id.
    """
    user = crud.user.get(db, id=user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    return user

@router.put("/{user_id}", response_model=schemas.User)
def update_user(
    *,
    db: Session = Depends(get_db),
    user_id: int,
    user_in: schemas.UserUpdate,
    current_user: models.User = Depends(get_current_active_superuser),
) -> Any:
    """
    Update a user.
    """
    user = crud.user.get(db, id=user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    user = crud.user.update(db, db_obj=user, obj_in=user_in)
    return user

@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(
    user_id: int,
    current_user: models.User = Depends(get_current_active_superuser),
    db: Session = Depends(get_db),
) -> Any:
    """
    Delete a user by ID.
    """
    user = crud.user.get(db, id=user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    db.delete(user)
    db.commit()
    return None

@router.post("/{user_id}/change-password", status_code=status.HTTP_200_OK)
def change_user_password(
    user_id: int,
    new_password: str = Body(..., embed=True),
    current_user: models.User = Depends(get_current_active_superuser),
    db: Session = Depends(get_db),
) -> Any:
    """
    Change a user's password.
    """
    user = crud.user.get(db, id=user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    user.hashed_password = models.User.get_password_hash(new_password)
    db.commit()
    db.refresh(user)
    return {"msg": "Password updated"}

@router.get("/{user_id}/activity", response_model=list)
def get_user_activity(
    user_id: int,
    current_user: models.User = Depends(get_current_active_superuser),
    db: Session = Depends(get_db),
) -> Any:
    """
    Get user activity log (stub).
    """
    # TODO: Replace with real activity log
    return [
        {"timestamp": "2024-01-01T00:00:00Z", "action": "login"},
        {"timestamp": "2024-01-02T12:00:00Z", "action": "trade"},
    ]
