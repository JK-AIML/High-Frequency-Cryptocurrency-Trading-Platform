"""
Load Testing for Tick Analysis System

This script performs load testing against the deployed application to validate
scaling behavior and performance characteristics.
"""

import time
import random
from locust import HttpUser, task, between, events
from locust.runners import MasterRunner, WorkerRunner

class TickAnalysisUser(HttpUser):
    """Simulates users interacting with the tick analysis system."""
    
    # Wait between 1 and 5 seconds between requests
    wait_time = between(1, 5)
    
    def on_start(self):
        """Called when a user starts."""
        self.client.verify = False  # Disable SSL verification for testing
    
    @task(5)
    def get_health(self):
        """Check service health."""
        self.client.get("/health")
    
    @task(3)
    def process_tick(self):
        """Process a tick."""
        tick = {
            "symbol": random.choice(["BTC-USD", "ETH-USD", "SOL-USD"]),
            "price": random.uniform(100, 100000),
            "volume": random.uniform(0.1, 100),
            "timestamp": int(time.time() * 1000),
            "exchange": random.choice(["binance", "coinbase", "kraken"])
        }
        self.client.post("/api/ticks", json=tick)
    
    @task(2)
    def get_metrics(self):
        """Get metrics."""
        self.client.get("/metrics")
    
    @task(1)
    def heavy_computation(self):
        """Trigger a heavy computation."""
        self.client.get("/api/compute?complexity=high")


def setup_test_users(environment, **kwargs):
    """Setup test data before the test starts."""
    if isinstance(environment.runner, MasterRunner):
        print("Setting up test users on master")
    elif isinstance(environment.runner, WorkerRunner):
        print(f"Setting up test users on worker {environment.runner.worker_index}")
    else:
        # For non-distributed mode
        print("Setting up test users")

# Hook for test setup
events.test_start.add_listener(setup_test_users)

# Example of custom event handlers
@events.request.add_listener
def on_request(request_type, name, response_time, response_length, response, context, exception, **kwargs):
    if exception:
        print(f"Request failed: {name} with exception {exception}")

@events.quitting.add_listener
def on_quitting(environment, **kwargs):
    """Called when the test is stopping."""
    print("Test is stopping")
    if environment.stats.total.fail_ratio > 0.1:
        print("Test failed due to high failure rate")
        environment.process_exit_code = 1
    elif environment.stats.total.avg_response_time > 2000:
        print("Test failed due to high response time")
        environment.process_exit_code = 1
    elif environment.stats.total.get("/api/ticks", 0).num_requests < 100:
        print("Test failed: Not enough requests")
        environment.process_exit_code = 1
    else:
        environment.process_exit_code = 0
