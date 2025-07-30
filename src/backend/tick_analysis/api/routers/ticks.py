"""
Tick data endpoints.
"""
from datetime import datetime, timedelta
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status, Body
from sqlalchemy.orm import Session

from ... import crud, models, schemas
from ...core.config import settings
from ...db.session import get_db
from ..deps import get_current_active_user

router = APIRouter(prefix="/ticks", tags=["ticks"])

@router.get("/", response_model=List[schemas.Tick])
def read_ticks(
    symbol: str = Query(..., description="Trading pair symbol (e.g., 'BTC-USD')"),
    start_time: Optional[datetime] = Query(
        None, 
        description="Start time for the query (default: 24 hours ago)"
    ),
    end_time: Optional[datetime] = Query(
        None, 
        description="End time for the query (default: now)"
    ),
    min_price: Optional[float] = Query(None, description="Minimum price"),
    max_price: Optional[float] = Query(None, description="Maximum price"),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_active_user),
) -> Any:
    """
    Retrieve tick data for a specific symbol within a time range and price range.
    """
    if end_time is None:
        end_time = datetime.utcnow()
    if start_time is None:
        start_time = end_time - timedelta(days=1)
    
    ticks = crud.tick.get_multi_by_symbol(
        db, 
        symbol=symbol,
        start_time=start_time,
        end_time=end_time
    )
    # Advanced filtering by price
    if min_price is not None:
        ticks = [t for t in ticks if t.price >= min_price]
    if max_price is not None:
        ticks = [t for t in ticks if t.price <= max_price]
    return ticks

@router.post("/", response_model=schemas.Tick, status_code=status.HTTP_201_CREATED)
def create_tick(
    *,
    tick_in: schemas.TickCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_active_user),
) -> Any:
    """
    Create new tick data point.
    """
    return crud.tick.create(db, obj_in=tick_in)

@router.get("/{tick_id}", response_model=schemas.Tick)
def read_tick(
    tick_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_active_user),
) -> Any:
    """
    Get a specific tick by ID.
    """
    tick = crud.tick.get(db, id=tick_id)
    if not tick:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tick not found",
        )
    return tick

@router.put("/{tick_id}", response_model=schemas.Tick)
def update_tick(
    tick_id: int,
    tick_in: schemas.TickUpdate = Body(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_active_user),
) -> Any:
    """
    Update a tick by ID.
    """
    tick = crud.tick.get(db, id=tick_id)
    if not tick:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tick not found")
    return crud.tick.update(db, db_obj=tick, obj_in=tick_in)

@router.delete("/{tick_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_tick(
    tick_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_active_user),
) -> Any:
    """
    Delete a tick by ID.
    """
    tick = crud.tick.get(db, id=tick_id)
    if not tick:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tick not found")
    crud.tick.remove(db, id=tick_id)
    return None
