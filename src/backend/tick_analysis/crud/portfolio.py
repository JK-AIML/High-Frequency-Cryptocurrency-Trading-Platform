"""
CRUD operations for portfolios.
"""
from typing import List, Optional, Dict, Any, Union
from datetime import datetime

from sqlalchemy.orm import Session
from sqlalchemy import and_

from ..db.models import Portfolio, User
from ..schemas.portfolio import PortfolioCreate, PortfolioUpdate

def get_portfolio(db: Session, portfolio_id: int) -> Optional[Portfolio]:
    """Get a portfolio by ID."""
    return db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()

def get_portfolios(
    db: Session, 
    user_id: int,
    skip: int = 0, 
    limit: int = 100,
    search: Optional[str] = None
) -> List[Portfolio]:
    """Get multiple portfolios for a user with optional search."""
    query = db.query(Portfolio).filter(Portfolio.user_id == user_id)
    
    if search:
        search = f"%{search}%"
        query = query.filter(Portfolio.name.ilike(search))
    
    return query.offset(skip).limit(limit).all()

def create_portfolio(
    db: Session, 
    portfolio_in: PortfolioCreate, 
    user_id: int
) -> Portfolio:
    """Create a new portfolio for a user."""
    db_portfolio = Portfolio(
        **portfolio_in.dict(),
        user_id=user_id
    )
    db.add(db_portfolio)
    db.commit()
    db.refresh(db_portfolio)
    return db_portfolio

def update_portfolio(
    db: Session, 
    db_portfolio: Portfolio, 
    portfolio_in: Union[PortfolioUpdate, Dict[str, Any]]
) -> Portfolio:
    """Update a portfolio."""
    portfolio_data = portfolio_in.dict(exclude_unset=True) if not isinstance(portfolio_in, dict) else portfolio_in
    
    for field, value in portfolio_data.items():
        setattr(db_portfolio, field, value)
    
    db_portfolio.updated_at = datetime.utcnow()
    db.add(db_portfolio)
    db.commit()
    db.refresh(db_portfolio)
    return db_portfolio

def delete_portfolio(db: Session, portfolio_id: int) -> Optional[Portfolio]:
    """Delete a portfolio."""
    db_portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
    if db_portfolio:
        db.delete(db_portfolio)
        db.commit()
    return db_portfolio

def get_user_portfolio(
    db: Session, 
    user_id: int, 
    portfolio_id: int
) -> Optional[Portfolio]:
    """Get a portfolio for a specific user."""
    return db.query(Portfolio).filter(
        and_(
            Portfolio.id == portfolio_id,
            Portfolio.user_id == user_id
        )
    ).first()
