"""Health check endpoints and monitoring for the streaming pipeline."""
from typing import Dict, Any, Optional
import psutil
import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime, timedelta

class HealthStatus(BaseModel):
    status: str
    version: str
    uptime: float
    timestamp: str
    details: Dict[str, Any]

class HealthMonitor:
    def __init__(self, app_name: str = "tick-analysis", version: str = "1.0.0"):
        self.app_name = app_name
        self.version = version
        self.start_time = time.time()
        self.checks = {}
        self.router = APIRouter()
        self._setup_routes()
    
    def _setup_routes(self):
        self.router.add_api_route("/health", self.health_check, methods=["GET"])
        self.router.add_api_route("/health/detailed", self.detailed_health, methods=["GET"])
    
    def register_check(self, name: str, check_func):
        """Register a health check function."""
        self.checks[name] = check_func
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Collect system-level health metrics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_used_mb": memory_info.rss / (1024 * 1024),
            "memory_percent": process.memory_percent(),
            "thread_count": process.num_threads(),
            "uptime": time.time() - self.start_time,
            "disk_usage": {d.mountpoint: {
                "total_gb": d.total / (1024**3),
                "used_gb": d.used / (1024**3),
                "free_gb": d.free / (1024**3),
                "percent_used": d.percent
            } for d in psutil.disk_partitions() if d.mountpoint}
        }
    
    async def health_check(self) -> HealthStatus:
        """Basic health check endpoint."""
        try:
            system_metrics = self.get_system_metrics()
            status = "healthy"
            details = {"system": system_metrics}
            
            # Run registered checks
            for name, check in self.checks.items():
                try:
                    result = await check() if hasattr(check, '__await__') else check()
                    details[name] = result
                    if result.get("status") != "healthy":
                        status = "degraded"
                except Exception as e:
                    details[name] = {"status": "unhealthy", "error": str(e)}
                    status = "unhealthy"
            
            return HealthStatus(
                status=status,
                version=self.version,
                uptime=system_metrics["uptime"],
                timestamp=datetime.utcnow().isoformat(),
                details=details
            )
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail={"status": "unhealthy", "error": str(e)}
            )
    
    async def detailed_health(self) -> Dict[str, Any]:
        """Detailed health check with system metrics and registered checks."""
        health = await self.health_check()
        return {
            **health.dict(),
            "system": self.get_system_metrics(),
            "checks": {name: check() for name, check in self.checks.items()}
        }
