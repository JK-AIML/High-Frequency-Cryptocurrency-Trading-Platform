"""
Prometheus metrics for monitoring the application.
"""
from prometheus_client import Counter, Histogram, Gauge

# Request metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total number of HTTP requests',
    ['method', 'endpoint', 'status_code']
)

REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency in seconds',
    ['endpoint']
)

# Error metrics
ERROR_COUNT = Counter(
    'http_errors_total',
    'Total number of HTTP errors',
    ['error_type']
)

# Cache metrics
CACHE_HITS = Counter(
    'cache_hits_total',
    'Total number of cache hits'
)

CACHE_MISSES = Counter(
    'cache_misses_total',
    'Total number of cache misses'
)

# Database metrics
DATABASE_QUERIES = Counter(
    'database_queries_total',
    'Total number of database queries',
    ['query_type']
)

# Application specific metrics
TICKS_PROCESSED = Counter(
    'ticks_processed_total',
    'Total number of tick data points processed'
)

# System metrics
SYSTEM_CPU_USAGE = Gauge(
    'system_cpu_usage_percent',
    'System CPU usage percentage'
)

SYSTEM_MEMORY_USAGE = Gauge(
    'system_memory_usage_bytes',
    'System memory usage in bytes'
)

def track_metrics(endpoint: str):
    """Decorator to track request metrics."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Track request start time
            start_time = time.time()
            
            try:
                # Call the endpoint
                response = await func(*args, **kwargs)
                
                # Track successful request
                duration = time.time() - start_time
                REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)
                REQUEST_COUNT.labels(
                    method=args[0].method if hasattr(args[0], 'method') else 'UNKNOWN',
                    endpoint=endpoint,
                    status_code=response.status_code
                ).inc()
                
                return response
                
            except Exception as e:
                # Track error
                ERROR_COUNT.labels(error_type=e.__class__.__name__).inc()
                raise
                
        return wrapper
    return decorator
