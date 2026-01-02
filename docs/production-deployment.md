# Production Deployment Guide

## Overview

This guide covers the complete production deployment of the SwellSight Wave Analysis Model, including MiDaS depth extraction, ControlNet synthetic generation, model versioning, API deployment, and monitoring setup.

## Prerequisites

### System Requirements
- **CPU**: 8+ cores recommended for production workloads
- **Memory**: 16GB+ RAM (32GB recommended for ControlNet generation)
- **GPU**: NVIDIA GPU with 8GB+ VRAM (optional but recommended for ControlNet)
- **Storage**: 100GB+ SSD for model artifacts and data
- **Network**: High-bandwidth connection for model downloads

### Software Dependencies
- Docker 20.10+
- Kubernetes 1.24+ (for container orchestration)
- Python 3.9+
- CUDA 11.8+ (if using GPU)

## Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone <repository-url>
cd swellsight

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Build Docker images
docker build -f docker/Dockerfile.inference -t swellsight:latest .
```

### 2. Basic Deployment

```bash
# Start with Docker Compose
docker-compose -f docker-compose.inference.yml up -d

# Verify deployment
curl http://localhost:8000/health
```

## MiDaS Depth Extraction Setup

### Installation

```bash
# Install MiDaS dependencies
pip install transformers torch torchvision
pip install opencv-python pillow numpy

# Download MiDaS model (automatic on first use)
python -c "from swellsight.data.midas_depth_extractor import MiDaSDepthExtractor; MiDaSDepthExtractor()"
```

### Configuration

```python
# MiDaS configuration in pipeline_config.json
{
  "midas_config": {
    "model_name": "Intel/dpt-large",  # or "Intel/dpt-hybrid-midas"
    "device": "cuda",  # or "cpu"
    "batch_size": 4,
    "quality_threshold": 0.5,
    "cache_depth_maps": true,
    "storage_format": "compressed_numpy"
  }
}
```

### Usage Examples

```python
from swellsight.data.midas_depth_extractor import MiDaSDepthExtractor

# Initialize extractor
extractor = MiDaSDepthExtractor(model_name="Intel/dpt-large")

# Extract depth from single image
depth_result = extractor.extract_depth("path/to/beach_image.jpg")

# Batch processing
depth_results = extractor.batch_extract([
    "image1.jpg", "image2.jpg", "image3.jpg"
])

# Quality validation
quality_score = extractor.validate_depth_quality(depth_result.depth_map)
```

## ControlNet Synthetic Generation

### Setup

```bash
# Install ControlNet dependencies
pip install diffusers transformers accelerate
pip install controlnet-aux

# Download ControlNet models (automatic on first use)
python -c "from swellsight.data.controlnet_generator import ControlNetSyntheticGenerator; ControlNetSyntheticGenerator()"
```

### Configuration

```python
# ControlNet configuration
{
  "controlnet_config": {
    "model_name": "lllyasviel/sd-controlnet-depth",
    "base_model": "runwayml/stable-diffusion-v1-5",
    "device": "cuda",
    "guidance_scale": 7.5,
    "num_inference_steps": 20,
    "controlnet_conditioning_scale": 1.0,
    "enable_memory_efficient_attention": true
  }
}
```

### Best Practices

1. **Prompt Engineering**:
   ```python
   # Good prompts for beach scenes
   positive_prompt = "photorealistic beach camera view, ocean waves, natural lighting, high quality"
   negative_prompt = "cartoon, anime, low quality, blurry, distorted"
   ```

2. **Memory Management**:
   ```python
   # Enable memory efficient attention for large images
   generator.enable_memory_efficient_attention()
   
   # Use gradient checkpointing
   generator.enable_gradient_checkpointing()
   ```

3. **Batch Processing**:
   ```python
   # Process in batches to manage memory
   batch_size = 2  # Adjust based on GPU memory
   for i in range(0, len(depth_maps), batch_size):
       batch = depth_maps[i:i+batch_size]
       synthetic_images = generator.batch_generate(batch, param_sets[i:i+batch_size])
   ```

## Augmentation Parameter System

### Configuration

The system supports 10 comprehensive augmentation categories:

```python
# Example augmentation configuration
{
  "augmentation_config": {
    "camera_view_geometry": {
      "height_range": [1, 50],  # meters
      "tilt_range": [-10, 30],  # degrees
      "fov_range": [30, 120],   # degrees
      "distance_range": [10, 500],  # meters
      "offset_range": [-100, 100]   # meters
    },
    "wave_field_structure": {
      "height_range": [0.3, 4.0],    # meters
      "wavelength_range": [5, 200],   # meters
      "period_range": [3, 20],        # seconds
      "spread_range": [0, 45],        # degrees
      "fronts_range": [1, 10]         # count
    },
    # ... additional categories
  }
}
```

### Parameter Validation

```python
from swellsight.data.augmentation_parameters import AugmentationParameterSystem

# Initialize parameter system
param_system = AugmentationParameterSystem()

# Generate and validate parameters
params = param_system.generate_parameters()
is_valid = param_system.validate_parameters(params)
plausibility_score = param_system.check_plausibility(params)
```

## Model Versioning and Registry

### Setup Model Registry

```python
from swellsight.utils.model_versioning import ModelVersionManager

# Initialize registry
registry = ModelVersionManager("./model_registry")

# Register new model version
version_info = registry.register_model(
    model_path="./checkpoints/model_v1.0.0.pth",
    version="1.0.0",
    metadata={
        "training_data": "synthetic_v2.1",
        "architecture": "convnext_base",
        "performance_metrics": {
            "height_mae": 0.45,
            "wave_type_accuracy": 0.82,
            "direction_accuracy": 0.87
        }
    }
)
```

### Model Deployment

```python
# Load specific model version
model_info = registry.get_model_version("1.0.0")
model = registry.load_model("1.0.0")

# A/B testing setup
registry.setup_ab_test("1.0.0", "1.1.0", traffic_split=0.1)
```

## Production API Deployment

### Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: swellsight-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: swellsight-api
  template:
    metadata:
      labels:
        app: swellsight-api
    spec:
      containers:
      - name: api
        image: swellsight:latest
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_VERSION
          value: "1.0.0"
        - name: DEVICE
          value: "cpu"
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

### Service Configuration

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: swellsight-service
spec:
  selector:
    app: swellsight-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Horizontal Pod Autoscaler

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: swellsight-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: swellsight-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## Monitoring and Observability

### Health Checks

```python
# Health check endpoints
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": get_model_version(),
        "uptime": get_uptime()
    }

@app.get("/ready")
async def readiness_check():
    # Check model loading, dependencies, etc.
    checks = {
        "model_loaded": model_manager.is_loaded(),
        "dependencies": check_dependencies(),
        "storage": check_storage_access()
    }
    
    if all(checks.values()):
        return {"status": "ready", "checks": checks}
    else:
        raise HTTPException(status_code=503, detail={"status": "not_ready", "checks": checks})
```

### Metrics Collection

```python
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
REQUEST_COUNT = Counter('swellsight_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('swellsight_request_duration_seconds', 'Request duration')
MODEL_PREDICTIONS = Counter('swellsight_predictions_total', 'Total predictions', ['model_version'])
ACTIVE_CONNECTIONS = Gauge('swellsight_active_connections', 'Active connections')

# Instrument endpoints
@REQUEST_DURATION.time()
def predict_wave_parameters(image_data):
    REQUEST_COUNT.labels(method='POST', endpoint='/predict').inc()
    MODEL_PREDICTIONS.labels(model_version=current_model_version).inc()
    # ... prediction logic
```

### Logging Configuration

```python
import logging
from pythonjsonlogger import jsonlogger

# Configure structured logging
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter(
    fmt='%(asctime)s %(name)s %(levelname)s %(message)s'
)
logHandler.setFormatter(formatter)

logger = logging.getLogger()
logger.addHandler(logHandler)
logger.setLevel(logging.INFO)

# Log prediction requests
logger.info("Prediction request", extra={
    "model_version": model_version,
    "image_size": image.size,
    "processing_time": processing_time,
    "prediction": prediction_summary
})
```

## Performance Optimization

### Model Optimization

```python
# Model quantization for CPU inference
import torch.quantization as quantization

# Post-training quantization
model_fp32 = load_model("model.pth")
model_int8 = quantization.quantize_dynamic(
    model_fp32, {torch.nn.Linear}, dtype=torch.qint8
)

# Save quantized model
torch.save(model_int8.state_dict(), "model_quantized.pth")
```

### Caching Strategy

```python
import redis
from functools import wraps

# Redis cache for predictions
redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cache_prediction(expiry=3600):
    def decorator(func):
        @wraps(func)
        def wrapper(image_hash, *args, **kwargs):
            cache_key = f"prediction:{image_hash}"
            cached_result = redis_client.get(cache_key)
            
            if cached_result:
                return json.loads(cached_result)
            
            result = func(image_hash, *args, **kwargs)
            redis_client.setex(cache_key, expiry, json.dumps(result))
            return result
        return wrapper
    return decorator
```

### Batch Processing

```python
from concurrent.futures import ThreadPoolExecutor
import asyncio

class BatchProcessor:
    def __init__(self, batch_size=8, timeout=1.0):
        self.batch_size = batch_size
        self.timeout = timeout
        self.pending_requests = []
        
    async def process_batch(self, requests):
        # Process multiple images in a single forward pass
        images = [req['image'] for req in requests]
        predictions = await self.model.batch_predict(images)
        
        return [
            {'request_id': req['id'], 'prediction': pred}
            for req, pred in zip(requests, predictions)
        ]
```

## Security Considerations

### API Security

```python
from fastapi import Security, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    token = credentials.credentials
    if not verify_jwt_token(token):
        raise HTTPException(status_code=401, detail="Invalid token")
    return token

@app.post("/predict")
async def predict(image: UploadFile, token: str = Depends(verify_token)):
    # Secure prediction endpoint
    pass
```

### Input Validation

```python
from PIL import Image
import magic

def validate_image(file_data: bytes) -> bool:
    # Check file type
    file_type = magic.from_buffer(file_data, mime=True)
    if file_type not in ['image/jpeg', 'image/png']:
        return False
    
    # Check image integrity
    try:
        image = Image.open(io.BytesIO(file_data))
        image.verify()
        return True
    except Exception:
        return False
```

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**:
   ```bash
   # Reduce batch size
   export BATCH_SIZE=1
   
   # Enable memory efficient attention
   export ENABLE_MEMORY_EFFICIENT_ATTENTION=true
   ```

2. **Model Loading Failures**:
   ```bash
   # Check model file integrity
   python -c "import torch; torch.load('model.pth')"
   
   # Verify model registry
   python -c "from swellsight.utils.model_versioning import ModelVersionManager; mgr = ModelVersionManager(); print(mgr.list_versions())"
   ```

3. **Slow Inference**:
   ```bash
   # Enable model quantization
   export USE_QUANTIZED_MODEL=true
   
   # Use GPU if available
   export DEVICE=cuda
   ```

### Monitoring Alerts

```yaml
# Prometheus alerting rules
groups:
- name: swellsight
  rules:
  - alert: HighErrorRate
    expr: rate(swellsight_requests_total{status="error"}[5m]) > 0.1
    for: 2m
    annotations:
      summary: "High error rate detected"
      
  - alert: SlowResponse
    expr: histogram_quantile(0.95, swellsight_request_duration_seconds) > 5
    for: 5m
    annotations:
      summary: "95th percentile response time > 5s"
      
  - alert: ModelAccuracyDrop
    expr: swellsight_model_accuracy < 0.7
    for: 10m
    annotations:
      summary: "Model accuracy dropped below 70%"
```

## Deployment Checklist

### Pre-deployment
- [ ] Model training completed and validated
- [ ] Model registered in version management system
- [ ] Performance benchmarks meet requirements
- [ ] Security review completed
- [ ] Load testing performed
- [ ] Monitoring and alerting configured

### Deployment
- [ ] Docker images built and pushed to registry
- [ ] Kubernetes manifests applied
- [ ] Health checks passing
- [ ] Metrics collection working
- [ ] Log aggregation configured

### Post-deployment
- [ ] Smoke tests passed
- [ ] Performance monitoring active
- [ ] Error rates within acceptable limits
- [ ] Model accuracy validation completed
- [ ] Rollback plan tested

## Support and Maintenance

### Regular Maintenance Tasks

1. **Model Updates**:
   - Monitor model performance metrics
   - Retrain models with new data
   - A/B test new model versions
   - Update model registry

2. **System Maintenance**:
   - Update dependencies
   - Security patches
   - Performance optimization
   - Capacity planning

3. **Data Management**:
   - Archive old predictions
   - Clean up temporary files
   - Backup model artifacts
   - Monitor storage usage

### Getting Help

- Check the [FAQ](faq.md) for common questions
- Review the [Troubleshooting Guide](troubleshooting.md) for known issues
- Monitor system logs and metrics
- Contact the development team for critical issues

## Conclusion

This guide provides comprehensive instructions for deploying the SwellSight Wave Analysis Model in production. Follow the deployment checklist and monitoring guidelines to ensure reliable operation.

For additional support, refer to the troubleshooting guide and FAQ documentation.