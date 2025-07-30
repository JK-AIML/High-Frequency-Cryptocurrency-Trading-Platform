"""
Database session management with async support.
"""
from typing import AsyncGenerator, Optional

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import QueuePool

from ..core.config import settings

# Create async engine and session factory
if settings.SQLALCHEMY_DATABASE_URI and settings.SQLALCHEMY_DATABASE_URI.startswith("postgresql"):
    # Convert to asyncpg URL format
    async_database_url = settings.SQLALCHEMY_DATABASE_URI.replace(
        "postgresql://", "postgresql+asyncpg://", 1
    )
else:
    # Fallback to SQLite for development
    async_database_url = "sqlite+aiosqlite:///./sql_app.db"

# Create async engine
engine = create_async_engine(
    async_database_url,
    echo=settings.DEBUG,
    future=True,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    pool_recycle=3600,
)

# Create session factory
async_session_factory = sessionmaker(
    engine, 
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# Base class for models
Base = declarative_base()

# Dependency
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency function that yields db sessions.
    
    Yields:
        AsyncSession: Database session
    """
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

async def init_db() -> None:
    """
    Initialize the database by creating tables.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def close_db() -> None:
    """
    Close the database connection.
    """
    await engine.dispose()