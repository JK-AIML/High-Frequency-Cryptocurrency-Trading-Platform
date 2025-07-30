"""
Pydantic schemas for request/response validation.
"""
from datetime import datetime
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, EmailStr, Field, validator

class Token(BaseModel):
    """JWT token schema."""
    access_token: str
    token_type: str

class TokenData(BaseModel):
    """Token data schema."""
    username: Optional[str] = None
    scopes: List[str] = []

class UserBase(BaseModel):
    """Base user schema."""
    email: EmailStr
    username: str
    full_name: Optional[str] = None
    is_active: Optional[bool] = True
    is_superuser: bool = False

class UserCreate(UserBase):
    """Schema for creating a user."""
    password: str = Field(..., min_length=8)
    
    @validator('password')
    def password_strength(cls, v):
        """Validate password strength."""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v

class UserUpdate(UserBase):
    """Schema for updating a user."""
    password: Optional[str] = Field(None, min_length=8)
    
    @validator('password')
    def password_strength(cls, v):
        """Validate password strength if provided."""
        if v is None:
            return v
        return UserCreate.password_strength.func(cls, v)  # type: ignore

class UserInDBBase(UserBase):
    """Base schema for user in database."""
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class User(UserInDBBase):
    """User schema for API responses."""
    pass

class PortfolioBase(BaseModel):
    """Base portfolio schema."""
    name: str
    description: Optional[str] = None

class PortfolioCreate(PortfolioBase):
    """Schema for creating a portfolio."""
    pass

class PortfolioUpdate(PortfolioBase):
    """Schema for updating a portfolio."""
    name: Optional[str] = None
    description: Optional[str] = None

class PortfolioInDBBase(PortfolioBase):
    """Base schema for portfolio in database."""
    id: int
    user_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class Portfolio(PortfolioInDBBase):
    """Portfolio schema for API responses."""
    pass

class PositionBase(BaseModel):
    """Base position schema."""
    symbol: str
    quantity: float
    entry_price: float

class PositionCreate(PositionBase):
    """Schema for creating a position."""
    pass

class PositionUpdate(BaseModel):
    """Schema for updating a position."""
    quantity: Optional[float] = None
    entry_price: Optional[float] = None

class PositionInDBBase(PositionBase):
    """Base schema for position in database."""
    id: int
    portfolio_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class Position(PositionInDBBase):
    """Position schema for API responses."""
    pass

class TickDataBase(BaseModel):
    """Base tick data schema."""
    symbol: str
    exchange: str
    price: float
    volume: float
    timestamp: datetime

class TickDataCreate(TickDataBase):
    """Schema for creating tick data."""
    pass

class TickDataInDBBase(TickDataBase):
    """Base schema for tick data in database."""
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

class TickData(TickDataInDBBase):
    """Tick data schema for API responses."""
    pass

# Response models
class ResponseModel(BaseModel):
    """Base response model with status and message."""
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None

class ErrorResponseModel(ResponseModel):
    """Error response model."""
    error: str
    details: Optional[Dict[str, Any]] = None
