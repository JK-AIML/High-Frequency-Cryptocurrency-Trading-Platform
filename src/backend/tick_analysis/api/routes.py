from fastapi import APIRouter, Depends, HTTPException, status, Body, Query
from sqlalchemy.orm import Session
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import json

from ..database import get_db
from ..models.base import User, Portfolio, Trade, Position, MarketData, TradingSignal
from ..services.api_service import get_api_service
from .main import get_current_user

router = APIRouter()

# API Connection Test endpoint
@router.get("/test-connections")
async def test_api_connections():
    """Test connections to all configured APIs."""
    api_service = get_api_service()
    results = await api_service.validate_connections()
    return {
        "status": "success",
        "connections": results,
        "timestamp": datetime.utcnow().isoformat()
    }

# Portfolio routes
@router.get("/portfolios/", response_model=List[dict])
async def get_portfolios(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    skip: int = Query(0, description="Number of records to skip for pagination"),
    limit: int = Query(100, description="Max records to return for pagination"),
):
    """Get all portfolios for the current user (paginated)"""
    portfolios = db.query(Portfolio).filter(Portfolio.user_id == current_user.id).offset(skip).limit(limit).all()
    return [
        {
            "id": p.id,
            "name": p.name,
            "created_at": p.created_at,
            "updated_at": p.updated_at
        }
        for p in portfolios
    ]

@router.post("/portfolios/", response_model=dict)
async def create_portfolio(
    name: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new portfolio"""
    portfolio = Portfolio(name=name, user_id=current_user.id)
    db.add(portfolio)
    db.commit()
    db.refresh(portfolio)
    return {
        "id": portfolio.id,
        "name": portfolio.name,
        "created_at": portfolio.created_at,
        "updated_at": portfolio.updated_at
    }

@router.get("/portfolios/{portfolio_id}", response_model=dict)
async def get_portfolio_by_id(
    portfolio_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get a single portfolio by ID"""
    portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id, Portfolio.user_id == current_user.id).first()
    if not portfolio:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Portfolio not found")
    return {
        "id": portfolio.id,
        "name": portfolio.name,
        "created_at": portfolio.created_at,
        "updated_at": portfolio.updated_at
    }

@router.put("/portfolios/{portfolio_id}", response_model=dict)
async def update_portfolio(
    portfolio_id: int,
    name: str = Body(..., embed=True),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update a portfolio (rename)"""
    portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id, Portfolio.user_id == current_user.id).first()
    if not portfolio:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Portfolio not found")
    portfolio.name = name
    db.commit()
    db.refresh(portfolio)
    return {
        "id": portfolio.id,
        "name": portfolio.name,
        "created_at": portfolio.created_at,
        "updated_at": portfolio.updated_at
    }

@router.delete("/portfolios/{portfolio_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_portfolio(
    portfolio_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a portfolio by ID"""
    portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id, Portfolio.user_id == current_user.id).first()
    if not portfolio:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Portfolio not found")
    db.delete(portfolio)
    db.commit()
    return None

@router.get("/portfolios/{portfolio_id}/metrics", response_model=dict)
async def get_portfolio_metrics(
    portfolio_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get portfolio performance/metrics (stub example)"""
    # TODO: Replace with real metrics logic
    portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id, Portfolio.user_id == current_user.id).first()
    if not portfolio:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Portfolio not found")
    # Example metrics
    return {
        "total_value": 100000,
        "total_return": 0.12,
        "sharpe_ratio": 1.5,
        "max_drawdown": 0.08,
        "positions_count": 5,
        "trades_count": 20
    }

# Market Data endpoint
@router.get("/market-data/{symbol}")
async def get_market_data(
    symbol: str,
    timeframe: str = "1m",
    current_user: User = Depends(get_current_user)
):
    """Get market data for a symbol from multiple sources."""
    api_service = get_api_service()
    data = await api_service.get_market_data(symbol, timeframe)
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "data": data,
        "timestamp": datetime.utcnow().isoformat()
    }

# Trading signals routes
@router.get("/signals/{symbol}")
async def get_trading_signals(
    symbol: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    db: Session = Depends(get_db)
):
    """Get trading signals for a symbol"""
    query = db.query(TradingSignal).filter(TradingSignal.symbol == symbol)
    
    if start_time:
        query = query.filter(TradingSignal.timestamp >= start_time)
    if end_time:
        query = query.filter(TradingSignal.timestamp <= end_time)
        
    signals = query.order_by(TradingSignal.timestamp).all()
    return [
        {
            "timestamp": s.timestamp,
            "signal_type": s.signal_type,
            "strength": s.strength,
            "source": s.source,
            "metadata": s.metadata
        }
        for s in signals
    ]

# Position routes
@router.get("/portfolios/{portfolio_id}/positions")
async def get_positions(
    portfolio_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all positions for a portfolio"""
    portfolio = db.query(Portfolio).filter(
        Portfolio.id == portfolio_id,
        Portfolio.user_id == current_user.id
    ).first()
    
    if not portfolio:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Portfolio not found"
        )
        
    positions = db.query(Position).filter(Position.portfolio_id == portfolio_id).all()
    return [
        {
            "symbol": p.symbol,
            "quantity": p.quantity,
            "average_price": p.average_price,
            "current_price": p.current_price,
            "last_updated": p.last_updated
        }
        for p in positions
    ]

# Trade routes
@router.get("/portfolios/{portfolio_id}/trades")
async def get_trades(
    portfolio_id: int,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    skip: int = Query(0, description="Number of records to skip for pagination"),
    limit: int = Query(100, description="Max records to return for pagination"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all trades for a portfolio (paginated, filterable)"""
    portfolio = db.query(Portfolio).filter(
        Portfolio.id == portfolio_id,
        Portfolio.user_id == current_user.id
    ).first()
    
    if not portfolio:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Portfolio not found"
        )
        
    query = db.query(Trade).filter(Trade.portfolio_id == portfolio_id)
    
    if start_time:
        query = query.filter(Trade.timestamp >= start_time)
    if end_time:
        query = query.filter(Trade.timestamp <= end_time)
    
    trades = query.order_by(Trade.timestamp).offset(skip).limit(limit).all()
    return [
        {
            "id": t.id,
            "symbol": t.symbol,
            "side": t.side,
            "quantity": t.quantity,
            "price": t.price,
            "timestamp": t.timestamp,
            "fees": t.fees
        }
        for t in trades
    ]

@router.get("/trades/{trade_id}")
async def get_trade_by_id(
    trade_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get a single trade by ID"""
    trade = db.query(Trade).filter(Trade.id == trade_id).first()
    if not trade:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Trade not found")
    # Optionally check user owns the portfolio
    return {
        "id": trade.id,
        "symbol": trade.symbol,
        "side": trade.side,
        "quantity": trade.quantity,
        "price": trade.price,
        "timestamp": trade.timestamp,
        "fees": trade.fees
    }

@router.put("/trades/{trade_id}")
async def update_trade(
    trade_id: int,
    updates: Dict = Body(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update a trade (e.g., add notes/tags)"""
    trade = db.query(Trade).filter(Trade.id == trade_id).first()
    if not trade:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Trade not found")
    for key, value in updates.items():
        if hasattr(trade, key):
            setattr(trade, key, value)
    db.commit()
    db.refresh(trade)
    return {
        "id": trade.id,
        "symbol": trade.symbol,
        "side": trade.side,
        "quantity": trade.quantity,
        "price": trade.price,
        "timestamp": trade.timestamp,
        "fees": trade.fees
    }

@router.delete("/trades/{trade_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_trade(
    trade_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a trade by ID"""
    trade = db.query(Trade).filter(Trade.id == trade_id).first()
    if not trade:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Trade not found")
    db.delete(trade)
    db.commit()
    return None 