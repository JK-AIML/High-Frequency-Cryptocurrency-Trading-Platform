import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from src.tick_analysis.database import Base, get_db
from src.tick_analysis.api.main import app
from fastapi.testclient import TestClient
from src.tick_analysis.models.base import User, Portfolio, Trade, Position, MarketData, TradingSignal
from passlib.context import CryptContext
import uuid
from datetime import datetime, timedelta
from jose import jwt

# Use in-memory SQLite database for testing
TEST_DATABASE_URL = "sqlite:///:memory:"

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

@pytest.fixture(scope="session")
def test_engine():
    """Create a test database engine for the whole test session."""
    engine = create_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)

@pytest.fixture(scope="function")
def test_db_session(test_engine):
    """Create a test database session with transaction rollback."""
    connection = test_engine.connect()
    transaction = connection.begin()
    TestingSessionLocal = sessionmaker(bind=connection)
    session = TestingSessionLocal()
    
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()

@pytest.fixture(scope="function")
def client(test_db_session):
    """Create a test client with the test database session."""
    app.dependency_overrides[get_db] = lambda: test_db_session
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()

@pytest.fixture(scope="function")
def test_user(test_db_session):
    """Create a test user in the database."""
    unique_username = f"testuser_{uuid.uuid4().hex[:8]}"
    user = User(
        username=unique_username,
        email=f"{unique_username}@example.com",
        hashed_password=pwd_context.hash("testpass123"),
        full_name="Test User",
        disabled=False,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    test_db_session.add(user)
    test_db_session.commit()
    test_db_session.refresh(user)
    return user

@pytest.fixture(scope="function")
def test_token(test_user):
    """Create a test JWT token."""
    SECRET_KEY = "test-secret-key"
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30
    
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode = {"sub": test_user.username, "exp": expire}
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

@pytest.fixture(scope="function")
def test_portfolio(test_db_session, test_user):
    """Create a test portfolio in the database."""
    portfolio = Portfolio(
        name="Test Portfolio",
        user_id=test_user.id,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    test_db_session.add(portfolio)
    test_db_session.commit()
    test_db_session.refresh(portfolio)
    return portfolio 