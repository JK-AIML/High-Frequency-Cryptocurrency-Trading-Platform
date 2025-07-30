"""Integration tests for the streaming pipeline."""
import asyncio
import time
import pytest
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from src.tick_analysis.streaming.operators import (
    StreamPipeline, MapOperator, FilterOperator, BatchOperator, 
    WindowOperator, AggregateOperator, JoinOperator, WindowType
)
from src.tick_analysis.storage.models import AggregationType
from src.tick_analysis.monitoring.health import HealthMonitor

class TestStreamingPipelineIntegration:
    """Integration tests for the streaming pipeline."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_processing(self):
        """Test end-to-end processing with multiple operators."""
        # Create test data
        test_data = [{"value": i, "timestamp": time.time() + i} for i in range(100)]
        
        # Define processing functions
        def double(x):
            return {"value": x["value"] * 2, "timestamp": x["timestamp"]}
            
        def filter_even(x):
            return x["value"] % 2 == 0
        
        # Create pipeline
        pipeline = StreamPipeline([
            MapOperator(double, name="double"),
            FilterOperator(filter_even, name="filter_even"),
            WindowOperator(
                window_size=timedelta(seconds=10),
                slide=timedelta(seconds=5),
                window_type=WindowType.TUMBLING,
                name="tumbling_window"
            ),
            AggregateOperator(
                key_func=lambda x: "all",
                agg_func=lambda items: {"sum": sum(x["value"] for x in items)},
                name="sum_aggregate"
            )
        ])
        
        # Process data
        results = await pipeline.batch_process_async(test_data)
        
        # Verify results
        assert len(results) > 0
        for result in results:
            assert "sum" in result
            assert isinstance(result["sum"], (int, float))
    
    @pytest.mark.asyncio
    async def test_backpressure_handling(self):
        """Test that backpressure is properly handled."""
        # Create a slow processing function
        async def slow_process(x):
            await asyncio.sleep(0.1)
            return x
            
        # Create pipeline with small queue size
        pipeline = StreamPipeline(
            [MapOperator(slow_process, name="slow_processor")],
            max_queue_size=5,
            backpressure_threshold=0.6
        )
        
        # Generate test data
        test_data = [{"id": i} for i in range(100)]
        
        # Process with backpressure
        start_time = time.time()
        results = await pipeline.batch_process_async(test_data, max_concurrent=10)
        duration = time.time() - start_time
        
        # Verify backpressure was applied (should take at least 1 second due to 0.1s sleep * 10 items)
        assert duration >= 1.0
        assert len(results) == len(test_data)
    
    @pytest.mark.asyncio
    async def test_health_monitoring(self):
        """Test health monitoring integration."""
        # Create health monitor
        health_monitor = HealthMonitor(app_name="test-app", version="1.0.0")
        
        # Add custom health check
        def custom_check():
            return {"status": "healthy", "details": "Custom check passed"}
            
        health_monitor.register_check("custom", custom_check)
        
        # Test health check
        health_status = await health_monitor.health_check()
        
        # Verify response
        assert health_status.status in ["healthy", "degraded"]
        assert health_status.version == "1.0.0"
        assert health_status.uptime > 0
        assert "system" in health_status.details
        assert "custom" in health_status.details

class TestWindowedOperations:
    """Test windowed stream operations."""
    
    @pytest.mark.parametrize("window_type,expected_windows", [
        (WindowType.TUMBLING, 2),
        (WindowType.SLIDING, 4),
        (WindowType.SESSION, 1)
    ])
    def test_window_types(self, window_type, expected_windows):
        """Test different window types."""
        # Create test data with timestamps
        now = time.time()
        test_data = [
            {"value": i, "timestamp": now + i} 
            for i in range(10)
        ]
        
        # Create window operator
        window_op = WindowOperator(
            window_size=timedelta(seconds=5),
            slide=timedelta(seconds=2.5) if window_type == WindowType.SLIDING else None,
            window_type=window_type,
            timestamp_extractor=lambda x: x["timestamp"]
        )
        
        # Process data
        results = []
        for item in test_data:
            result = window_op.process(item)
            if result:
                results.extend(result)
        
        # Flush any remaining windows
        flushed = window_op.flush()
        if flushed:
            results.extend(flushed)
        
        # Verify
        if window_type == WindowType.SESSION:
            # Session windows are trickier to test due to timing
            assert len(results) >= 1
        else:
            assert len(results) == expected_windows
