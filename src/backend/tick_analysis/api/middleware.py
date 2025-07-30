"""
Custom middleware for the FastAPI application.
"""
import time
from typing import Callable, Awaitable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from ..core.metrics import (
    REQUEST_LATENCY,
    REQUEST_COUNT,
    ERROR_COUNT
)

class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting request metrics."""
    
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # Skip metrics endpoint
        if request.url.path == "/metrics":
            return await call_next(request)
            
        start_time = time.time()
        status_code = 500
        
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception as e:
            status_code = getattr(e, 'status_code', 500)
            ERROR_COUNT.labels(error_type=e.__class__.__name__).inc()
            raise
        finally:
            duration = time.time() - start_time
            REQUEST_LATENCY.labels(endpoint=request.url.path).observe(duration)
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                status_code=status_code
            ).inc()

class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging."""
    
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # Log request
        print(f"Request: {request.method} {request.url}")
        
        # Process request
        response = await call_next(request)
        
        # Log response
        print(f"Response: {response.status_code}")
        
        return response
