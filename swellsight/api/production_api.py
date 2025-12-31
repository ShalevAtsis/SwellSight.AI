"""Production API for SwellSight Wave Analysis Model."""

import asyncio
import time
import logging
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
import hashlib
import uuid

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
import torch
from PIL import Image
import io

from ..inference.inference_engine import InferenceEngine, WavePrediction, InferenceError
from ..utils.model_versioning import ModelVersionManager
from ..utils.performance_benchmarks import PerformanceBenchmark
from ..config import ModelConfig


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

# Security
security = HTTPBearer()


@dataclass
class APIMetrics:
    """API performance metrics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_response_time: float = 0.0
    min_response_time: float = float('inf')
    max_response_time: float = 0.0
    error_counts: Dict[str, int] = None
    
    def __post_init__(self):
        if self.error_counts is None:
            self.error_counts = {}
    
    @property
    def average_response_time(self) -> float:
        """Calculate average response time."""
        if self.total_requests == 0:
            return 0.0
        return self.total_response_time / self.total_requests
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests


class APIMonitor:
    """API monitoring and metrics collection."""
    
    def __init__(self):
        """Initialize API monitor."""
        self.metrics = APIMetrics()
        self.request_history: List[Dict[str, Any]] = []
        self.max_history_size = 10000
        self.performance_thresholds = {
            "max_response_time": 2.0,  # seconds
            "min_success_rate": 0.95,  # 95%
            "max_error_rate": 0.05     # 5%
        }
        self.alerts: List[Dict[str, Any]] = []
    
    def record_request(
        self,
        request_id: str,
        endpoint: str,
        method: str,
        response_time: float,
        status_code: int,
        error_type: Optional[str] = None
    ):
        """Record API request metrics."""
        # Update metrics
        self.metrics.total_requests += 1
        self.metrics.total_response_time += response_time
        self.metrics.min_response_time = min(self.metrics.min_response_time, response_time)
        self.metrics.max_response_time = max(self.metrics.max_response_time, response_time)
        
        if 200 <= status_code < 300:
            self.metrics.successful_requests += 1
        else:
            self.metrics.failed_requests += 1
            if error_type:
                self.metrics.error_counts[error_type] = self.metrics.error_counts.get(error_type, 0) + 1
        
        # Record request history
        request_record = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "endpoint": endpoint,
            "method": method,
            "response_time": response_time,
            "status_code": status_code,
            "error_type": error_type
        }
        
        self.request_history.append(request_record)
        
        # Limit history size
        if len(self.request_history) > self.max_history_size:
            self.request_history = self.request_history[-self.max_history_size:]
        
        # Check for performance issues
        self._check_performance_alerts(response_time, status_code)
    
    def _check_performance_alerts(self, response_time: float, status_code: int):
        """Check for performance issues and generate alerts."""
        alerts = []
        
        # Check response time
        if response_time > self.performance_thresholds["max_response_time"]:
            alerts.append({
                "type": "high_response_time",
                "message": f"Response time {response_time:.2f}s exceeds threshold {self.performance_thresholds['max_response_time']}s",
                "timestamp": datetime.now().isoformat(),
                "severity": "warning"
            })
        
        # Check success rate (over last 100 requests)
        recent_requests = self.request_history[-100:] if len(self.request_history) >= 100 else self.request_history
        if recent_requests:
            recent_success_rate = sum(1 for r in recent_requests if 200 <= r["status_code"] < 300) / len(recent_requests)
            if recent_success_rate < self.performance_thresholds["min_success_rate"]:
                alerts.append({
                    "type": "low_success_rate",
                    "message": f"Success rate {recent_success_rate:.2%} below threshold {self.performance_thresholds['min_success_rate']:.2%}",
                    "timestamp": datetime.now().isoformat(),
                    "severity": "critical"
                })
        
        # Add alerts
        self.alerts.extend(alerts)
        
        # Limit alerts history
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-1000:]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current API metrics."""
        return {
            "metrics": asdict(self.metrics),
            "recent_alerts": self.alerts[-10:],  # Last 10 alerts
            "timestamp": datetime.now().isoformat()
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get API health status."""
        # Check recent performance
        recent_requests = self.request_history[-100:] if len(self.request_history) >= 100 else self.request_history
        
        if not recent_requests:
            return {
                "status": "unknown",
                "message": "No recent requests to analyze",
                "timestamp": datetime.now().isoformat()
            }
        
        recent_success_rate = sum(1 for r in recent_requests if 200 <= r["status_code"] < 300) / len(recent_requests)
        recent_avg_response_time = sum(r["response_time"] for r in recent_requests) / len(recent_requests)
        
        # Determine health status
        if (recent_success_rate >= self.performance_thresholds["min_success_rate"] and 
            recent_avg_response_time <= self.performance_thresholds["max_response_time"]):
            status = "healthy"
            message = "API is performing within acceptable parameters"
        elif recent_success_rate < 0.8 or recent_avg_response_time > 5.0:
            status = "unhealthy"
            message = f"API performance degraded: success_rate={recent_success_rate:.2%}, avg_response_time={recent_avg_response_time:.2f}s"
        else:
            status = "degraded"
            message = f"API performance below optimal: success_rate={recent_success_rate:.2%}, avg_response_time={recent_avg_response_time:.2f}s"
        
        return {
            "status": status,
            "message": message,
            "metrics": {
                "recent_success_rate": recent_success_rate,
                "recent_avg_response_time": recent_avg_response_time,
                "total_requests": len(recent_requests)
            },
            "timestamp": datetime.now().isoformat()
        }


# Pydantic models for API
class WavePredictionResponse(BaseModel):
    """Wave prediction API response."""
    request_id: str
    height_meters: float = Field(..., description="Predicted wave height in meters")
    wave_type: str = Field(..., description="Predicted wave breaking type")
    direction: str = Field(..., description="Predicted wave direction")
    wave_type_probabilities: Dict[str, float] = Field(..., description="Wave type classification probabilities")
    direction_probabilities: Dict[str, float] = Field(..., description="Direction classification probabilities")
    confidence_scores: Dict[str, float] = Field(..., description="Confidence scores for each prediction")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    model_version: str = Field(..., description="Model version used for prediction")
    timestamp: str = Field(..., description="Prediction timestamp")


class ErrorResponse(BaseModel):
    """API error response."""
    error: str
    message: str
    request_id: str
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    message: str
    metrics: Optional[Dict[str, Any]] = None
    timestamp: str


class MetricsResponse(BaseModel):
    """Metrics response."""
    metrics: Dict[str, Any]
    recent_alerts: List[Dict[str, Any]]
    timestamp: str


# Global instances
monitor = APIMonitor()
inference_engine: Optional[InferenceEngine] = None
version_manager: Optional[ModelVersionManager] = None


def get_inference_engine() -> InferenceEngine:
    """Get inference engine dependency."""
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")
    return inference_engine


def get_version_manager() -> ModelVersionManager:
    """Get version manager dependency."""
    if version_manager is None:
        raise HTTPException(status_code=503, detail="Version manager not initialized")
    return version_manager


async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify authentication token."""
    # Simple token verification - in production, use proper JWT validation
    token = credentials.credentials
    
    # For demo purposes, accept any non-empty token
    # In production, implement proper token validation
    if not token or len(token) < 10:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return token


def create_app(
    model_path: Optional[str] = None,
    registry_path: str = "models/registry",
    enable_auth: bool = False,
    cors_origins: List[str] = None
) -> FastAPI:
    """Create FastAPI application."""
    
    app = FastAPI(
        title="SwellSight Wave Analysis API",
        description="Production API for wave parameter prediction from beach camera images",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add middleware
    app.add_middleware(SlowAPIMiddleware)
    
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]  # Configure appropriately for production
    )
    
    # Add rate limiting error handler
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize services on startup."""
        global inference_engine, version_manager
        
        try:
            # Initialize version manager
            version_manager = ModelVersionManager(registry_path)
            logger.info(f"Version manager initialized with registry: {registry_path}")
            
            # Initialize inference engine
            if model_path:
                # Load specific model
                from ..utils.model_persistence import load_model
                model = load_model(model_path)
                config = ModelConfig()  # Use default config
                inference_engine = InferenceEngine(model, config)
                logger.info(f"Inference engine initialized with model: {model_path}")
            else:
                # Load latest model from registry
                model, model_version = version_manager.get_model()
                config = ModelConfig()  # Use default config
                inference_engine = InferenceEngine(model, config)
                logger.info(f"Inference engine initialized with latest model: {model_version.version}")
            
        except Exception as e:
            logger.error(f"Failed to initialize services: {e}")
            raise
    
    @app.middleware("http")
    async def monitor_requests(request: Request, call_next):
        """Monitor all requests for metrics collection."""
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        start_time = time.time()
        
        try:
            response = await call_next(request)
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Record metrics
            monitor.record_request(
                request_id=request_id,
                endpoint=str(request.url.path),
                method=request.method,
                response_time=response_time,
                status_code=response.status_code
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            
            # Record error metrics
            monitor.record_request(
                request_id=request_id,
                endpoint=str(request.url.path),
                method=request.method,
                response_time=response_time,
                status_code=500,
                error_type=type(e).__name__
            )
            
            raise
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        health_status = monitor.get_health_status()
        return HealthResponse(**health_status)
    
    @app.get("/metrics", response_model=MetricsResponse)
    async def get_metrics(
        token: str = Depends(verify_token) if enable_auth else None
    ):
        """Get API metrics (requires authentication if enabled)."""
        metrics_data = monitor.get_metrics()
        return MetricsResponse(**metrics_data)
    
    @app.get("/model/info")
    async def get_model_info(
        engine: InferenceEngine = Depends(get_inference_engine)
    ):
        """Get information about the current model."""
        model_info = engine.get_model_info()
        return {
            "model_info": model_info,
            "timestamp": datetime.now().isoformat()
        }
    
    @app.get("/model/versions")
    async def list_model_versions(
        vm: ModelVersionManager = Depends(get_version_manager),
        token: str = Depends(verify_token) if enable_auth else None
    ):
        """List available model versions (requires authentication if enabled)."""
        versions = vm.list_versions()
        return {
            "versions": [asdict(v) for v in versions],
            "latest": vm.registry.get("latest"),
            "timestamp": datetime.now().isoformat()
        }
    
    @app.post("/predict", response_model=WavePredictionResponse)
    @limiter.limit("100/minute")  # Rate limit: 100 requests per minute
    async def predict_wave_parameters(
        request: Request,
        background_tasks: BackgroundTasks,
        image: UploadFile = File(..., description="Beach camera image (JPEG or PNG)"),
        engine: InferenceEngine = Depends(get_inference_engine)
    ):
        """
        Predict wave parameters from uploaded image.
        
        - **image**: Beach camera image file (JPEG or PNG format)
        - Returns wave height, type, direction with confidence scores
        """
        request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))
        start_time = time.time()
        
        try:
            # Validate file type
            if not image.content_type or not image.content_type.startswith('image/'):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file type: {image.content_type}. Expected image file."
                )
            
            # Read and validate image
            image_data = await image.read()
            if len(image_data) == 0:
                raise HTTPException(status_code=400, detail="Empty image file")
            
            if len(image_data) > 10 * 1024 * 1024:  # 10MB limit
                raise HTTPException(status_code=400, detail="Image file too large (max 10MB)")
            
            # Convert to PIL Image for validation
            try:
                pil_image = Image.open(io.BytesIO(image_data))
                pil_image.verify()  # Verify image integrity
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
            
            # Create temporary file for inference
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                temp_file.write(image_data)
                temp_path = temp_file.name
            
            try:
                # Run inference
                prediction = engine.predict(temp_path)
                
                processing_time = (time.time() - start_time) * 1000  # Convert to ms
                
                # Create response
                response = WavePredictionResponse(
                    request_id=request_id,
                    height_meters=prediction.height_meters,
                    wave_type=prediction.wave_type,
                    direction=prediction.direction,
                    wave_type_probabilities=prediction.wave_type_probs,
                    direction_probabilities=prediction.direction_probs,
                    confidence_scores=prediction.confidence_scores,
                    processing_time_ms=processing_time,
                    model_version="latest",  # TODO: Get actual version from engine
                    timestamp=datetime.now().isoformat()
                )
                
                # Schedule cleanup
                background_tasks.add_task(cleanup_temp_file, temp_path)
                
                return response
                
            finally:
                # Ensure cleanup even if inference fails
                background_tasks.add_task(cleanup_temp_file, temp_path)
        
        except HTTPException:
            raise
        except InferenceError as e:
            raise HTTPException(status_code=422, detail=f"Inference failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in prediction: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")
    
    @app.post("/predict/batch")
    @limiter.limit("10/minute")  # Lower rate limit for batch processing
    async def predict_batch(
        request: Request,
        background_tasks: BackgroundTasks,
        images: List[UploadFile] = File(..., description="List of beach camera images"),
        engine: InferenceEngine = Depends(get_inference_engine),
        token: str = Depends(verify_token) if enable_auth else None
    ):
        """
        Predict wave parameters for multiple images (requires authentication if enabled).
        
        - **images**: List of beach camera image files
        - Returns list of predictions for each image
        """
        request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))
        
        if len(images) > 10:  # Limit batch size
            raise HTTPException(status_code=400, detail="Batch size limited to 10 images")
        
        results = []
        temp_files = []
        
        try:
            # Process each image
            for i, image in enumerate(images):
                start_time = time.time()
                
                # Validate and save image
                image_data = await image.read()
                
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                    temp_file.write(image_data)
                    temp_path = temp_file.name
                    temp_files.append(temp_path)
                
                try:
                    # Run inference
                    prediction = engine.predict(temp_path)
                    processing_time = (time.time() - start_time) * 1000
                    
                    result = {
                        "image_index": i,
                        "filename": image.filename,
                        "prediction": prediction.to_dict(),
                        "processing_time_ms": processing_time
                    }
                    results.append(result)
                    
                except Exception as e:
                    results.append({
                        "image_index": i,
                        "filename": image.filename,
                        "error": str(e),
                        "processing_time_ms": (time.time() - start_time) * 1000
                    })
            
            # Schedule cleanup for all temp files
            for temp_path in temp_files:
                background_tasks.add_task(cleanup_temp_file, temp_path)
            
            return {
                "request_id": request_id,
                "results": results,
                "total_images": len(images),
                "successful_predictions": len([r for r in results if "prediction" in r]),
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            # Cleanup on error
            for temp_path in temp_files:
                background_tasks.add_task(cleanup_temp_file, temp_path)
            raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")
    
    return app


async def cleanup_temp_file(file_path: str):
    """Clean up temporary file."""
    try:
        Path(file_path).unlink(missing_ok=True)
    except Exception as e:
        logger.warning(f"Failed to cleanup temp file {file_path}: {e}")


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    model_path: Optional[str] = None,
    registry_path: str = "models/registry",
    enable_auth: bool = False,
    cors_origins: List[str] = None,
    workers: int = 1
):
    """Run the production API server."""
    app = create_app(
        model_path=model_path,
        registry_path=registry_path,
        enable_auth=enable_auth,
        cors_origins=cors_origins
    )
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        workers=workers,
        log_level="info",
        access_log=True
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SwellSight Production API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--model-path", help="Path to specific model file")
    parser.add_argument("--registry-path", default="models/registry", help="Path to model registry")
    parser.add_argument("--enable-auth", action="store_true", help="Enable authentication")
    parser.add_argument("--cors-origins", nargs="*", help="CORS allowed origins")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    
    args = parser.parse_args()
    
    run_server(
        host=args.host,
        port=args.port,
        model_path=args.model_path,
        registry_path=args.registry_path,
        enable_auth=args.enable_auth,
        cors_origins=args.cors_origins,
        workers=args.workers
    )