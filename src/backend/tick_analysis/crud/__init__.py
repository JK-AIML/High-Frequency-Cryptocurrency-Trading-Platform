"""
CRUD (Create, Read, Update, Delete) operations for the application.
"""
from .user import (
    get_user,
    get_user_by_email,
    get_user_by_username,
    get_users,
    create_user,
    update_user,
    delete_user,
    authenticate,
    is_active,
    is_superuser
)

from .portfolio import (
    get_portfolio,
    get_portfolios,
    create_portfolio,
    update_portfolio,
    delete_portfolio,
    get_user_portfolio
)

from .tick import (
    get_tick,
    get_ticks,
    create_tick,
    create_ticks_bulk,
    get_latest_tick,
    get_ohlcv
)

__all__ = [
    # User
    'get_user',
    'get_user_by_email',
    'get_user_by_username',
    'get_users',
    'create_user',
    'update_user',
    'delete_user',
    'authenticate',
    'is_active',
    'is_superuser',
    
    # Portfolio
    'get_portfolio',
    'get_portfolios',
    'create_portfolio',
    'update_portfolio',
    'delete_portfolio',
    'get_user_portfolio',
    
    # Tick Data
    'get_tick',
    'get_ticks',
    'create_tick',
    'create_ticks_bulk',
    'get_latest_tick',
    'get_ohlcv'
]
