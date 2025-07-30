"""
CRUD operations for tick data.
"""
from typing import List, Optional, Dict, Any, Union, Tuple
from datetime import datetime, timedelta

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func
import pandas as pd

from ..db.models import TickData
from ..schemas.tick import TickDataCreate, TickDataUpdate, TickDataInDB

def get_tick(db: Session, tick_id: int) -> Optional[TickData]:
    """Get a tick by ID."""
    return db.query(TickData).filter(TickData.id == tick_id).first()

def get_ticks(
    db: Session,
    symbol: Optional[str] = None,
    exchange: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    skip: int = 0,
    limit: int = 1000
) -> List[TickData]:
    """
    Get multiple ticks with optional filtering.
    
    Args:
        db: Database session
        symbol: Filter by symbol
        exchange: Filter by exchange
        start_time: Filter ticks after this time
        end_time: Filter ticks before this time
        skip: Number of records to skip
        limit: Maximum number of records to return
        
    Returns:
        List of tick data records
    """
    query = db.query(TickData)
    
    if symbol:
        query = query.filter(TickData.symbol == symbol)
    if exchange:
        query = query.filter(TickData.exchange == exchange)
    if start_time:
        query = query.filter(TickData.timestamp >= start_time)
    if end_time:
        query = query.filter(TickData.timestamp <= end_time)
    
    return query.order_by(TickData.timestamp).offset(skip).limit(limit).all()

def create_tick(db: Session, tick_in: TickDataCreate) -> TickData:
    """Create a new tick data record."""
    db_tick = TickData(**tick_in.dict())
    db.add(db_tick)
    db.commit()
    db.refresh(db_tick)
    return db_tick

def create_ticks_bulk(db: Session, ticks: List[Dict[str, Any]]) -> int:
    """
    Create multiple tick data records in bulk.
    
    Args:
        db: Database session
        ticks: List of tick data dictionaries
        
    Returns:
        Number of records created
    """
    if not ticks:
        return 0
        
    db.bulk_insert_mappings(TickData, ticks)
    db.commit()
    return len(ticks)

def get_latest_tick(db: Session, symbol: str, exchange: str) -> Optional[TickData]:
    """Get the most recent tick for a symbol and exchange."""
    return db.query(TickData).filter(
        and_(
            TickData.symbol == symbol,
            TickData.exchange == exchange
        )
    ).order_by(TickData.timestamp.desc()).first()

def get_ohlcv(
    db: Session,
    symbol: str,
    exchange: str,
    interval: str = '1m',
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = 1000
) -> List[Dict[str, Any]]:
    """
    Get OHLCV (Open, High, Low, Close, Volume) data for a symbol.
    
    Args:
        db: Database session
        symbol: Symbol to get data for
        exchange: Exchange name
        interval: Time interval (e.g., '1m', '1h', '1d')
        start_time: Start time for the query
        end_time: End time for the query
        limit: Maximum number of candles to return
        
    Returns:
        List of OHLCV data points
    """
    # Convert interval to timedelta
    interval_map = {
        '1m': timedelta(minutes=1),
        '5m': timedelta(minutes=5),
        '15m': timedelta(minutes=15),
        '1h': timedelta(hours=1),
        '4h': timedelta(hours=4),
        '1d': timedelta(days=1)
    }
    
    interval_td = interval_map.get(interval, timedelta(minutes=1))
    
    # Build the base query
    query = db.query(
        func.date_trunc(interval, TickData.timestamp).label('timestamp'),
        func.first_value(TickData.price).over(
            partition_by=func.date_trunc(interval, TickData.timestamp),
            order_by=TickData.timestamp
        ).label('open'),
        func.max(TickData.price).label('high'),
        func.min(TickData.price).label('low'),
        func.last_value(TickData.price).over(
            partition_by=func.date_trunc(interval, TickData.timestamp),
            order_by=TickData.timestamp
        ).label('close'),
        func.sum(TickData.volume).label('volume'),
        func.count().label('tick_count')
    ).filter(
        and_(
            TickData.symbol == symbol,
            TickData.exchange == exchange
        )
    )
    
    # Apply time filters
    if start_time:
        query = query.filter(TickData.timestamp >= start_time)
    if end_time:
        query = query.filter(TickData.timestamp <= end_time)
    
    # Group by time interval
    query = query.group_by(
        func.date_trunc(interval, TickData.timestamp)
    ).order_by(
        func.date_trunc(interval, TickData.timestamp).desc()
    ).limit(limit)
    
    # Execute query and format results
    results = query.all()
    return [
        {
            'timestamp': r.timestamp,
            'open': float(r.open),
            'high': float(r.high),
            'low': float(r.low),
            'close': float(r.close),
            'volume': float(r.volume) if r.volume else 0.0,
            'tick_count': r.tick_count
        }
        for r in results
    ]
