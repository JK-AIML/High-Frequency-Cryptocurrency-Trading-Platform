import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from src.tick_analysis.api.main import app
from src.tick_analysis.database import get_db
from src.tick_analysis.models.base import User, Portfolio
from passlib.context import CryptContext
import asyncio
from datetime import datetime, timedelta
from jose import jwt
import uuid

# Test configuration
SECRET_KEY = "test-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Test data
TEST_USER = {
    "username": "testuser",
    "email": "test@example.com",
    "password": "testpass123",
    "full_name": "Test User"
}

def create_test_token():
    """Create a test JWT token."""
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode = {"sub": TEST_USER["username"], "exp": expire}
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest_asyncio.fixture(scope="function")
async def test_client(test_db_session):
    """Create an async test client with the test database session."""
    app.dependency_overrides[get_db] = lambda: test_db_session
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    app.dependency_overrides.clear()

# Basic endpoint tests
@pytest.mark.asyncio
async def test_health_check(test_client):
    response = await test_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

@pytest.mark.asyncio
async def test_root(test_client):
    response = await test_client.get("/")
    assert response.status_code == 200
    assert "Welcome" in response.json()["message"]

# Authentication tests
@pytest.mark.asyncio
async def test_login_success(test_client, test_user):
    response = await test_client.post(
        "/token",
        data={
            "username": test_user.username,
            "password": "testpass123"
        }
    )
    assert response.status_code == 200
    assert "access_token" in response.json()
    assert response.json()["token_type"] == "bearer"

@pytest.mark.asyncio
async def test_login_invalid_credentials(test_client):
    response = await test_client.post(
        "/token",
        data={
            "username": "wronguser",
            "password": "wrongpass"
        }
    )
    assert response.status_code == 401

# Market data tests
@pytest.mark.asyncio
async def test_market_data_unauthorized(test_client):
    response = await test_client.get("/market-data/BTC/USD")
    assert response.status_code == 401

@pytest.mark.asyncio
async def test_market_data_authorized(test_client, test_token):
    response = await test_client.get(
        "/market-data/BTC/USD",
        headers={"Authorization": f"Bearer {test_token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "symbol" in data
    assert "data" in data
    assert "timestamp" in data

# Portfolio tests
@pytest.mark.asyncio
async def test_create_portfolio(test_client, test_token):
    response = await test_client.post(
        "/portfolios/",
        headers={"Authorization": f"Bearer {test_token}"},
        json={"name": "Test Portfolio"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert data["name"] == "Test Portfolio"

@pytest.mark.asyncio
async def test_get_portfolios(test_client, test_token):
    response = await test_client.get(
        "/portfolios/",
        headers={"Authorization": f"Bearer {test_token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    if data:
        assert "id" in data[0]
        assert "name" in data[0]

# API connection test
@pytest.mark.asyncio
async def test_api_connections(test_client, test_token):
    response = await test_client.get(
        "/test-connections",
        headers={"Authorization": f"Bearer {test_token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "connections" in data
    assert "timestamp" in data

# Error handling tests
@pytest.mark.asyncio
async def test_invalid_endpoint(test_client):
    response = await test_client.get("/invalid-endpoint")
    assert response.status_code == 404

@pytest.mark.asyncio
async def test_invalid_token(test_client):
    response = await test_client.get(
        "/portfolios/",
        headers={"Authorization": "Bearer invalid-token"}
    )
    assert response.status_code == 401

@pytest.mark.asyncio
async def test_docs(test_client):
    response = await test_client.get("/docs")
    assert response.status_code == 200
    assert "Swagger UI" in response.text

@pytest.mark.asyncio
async def test_openapi(test_client):
    response = await test_client.get("/openapi.json")
    assert response.status_code == 200
    assert response.json()["info"]["title"] == "Tick Analysis API" 