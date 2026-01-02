# Troubleshooting Guide

## Overview

This guide covers common issues, error messages, and solutions for the SwellSight Wave Analysis Model deployment and operation.

## Common Issues

### 1. Model Loading Issues

#### Error: "Model file not found"
```
FileNotFoundError: [Errno 2] No such file or directory: 'model.pth'
```

**Causes:**
- Model file path is incorrect
- Model not properly saved during training
- File permissions issue

**Solutions:**
```bash
# Check if model file exists
ls -la checkpoints/

# Verify model registry
python -c "from swellsight.utils.model_versioning import ModelVersionManager; mgr = ModelVersionManager(); print(mgr.list_versions())"

# Check file permissions
chmod 644 checkpoints/*.pth

# Re-download or retrain model if necessary
python swellsight/scripts/train_end_to_end.py --config config.json
```

#### Error: "Model architecture mismatch"
```
RuntimeError: Error(s) in loading state_dict for WaveAnalysisModel
```

**Causes:**
- Model architecture changed between training and inference
- Incorrect model configuration
- Corrupted model file

**Solutions:**
```python
# Verify model configuration matches training
from swellsight.config import ModelConfig
config = ModelConfig.from_file("model_config.json")
print(config)

# Check model integrity
import torch
try:
    checkpoint = torch.load("model.pth", map_location='cpu')
    print("Model loaded successfully")
    print("Keys:", list(checkpoint.keys()))
except Exception as e:
    print(f"Model loading failed: {e}")

# Rebuild model with correct configuration
from swellsight.models import WaveAnalysisModel
model = WaveAnalysisModel(config)
model.load_state_dict(checkpoint['model_state_dict'])
```

### 2. Memory Issues

#### Error: "CUDA out of memory"
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Causes:**
- Batch size too large for available GPU memory
- Model too large for GPU
- Memory leak in processing pipeline

**Solutions:**
```bash
# Reduce batch size
export BATCH_SIZE=1

# Enable memory efficient attention
export ENABLE_MEMORY_EFFICIENT_ATTENTION=true

# Use CPU instead of GPU
export DEVICE=cpu

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"
```

```python
# Monitor GPU memory usage
import torch
if torch.cuda.is_available():
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
```

#### Error: "System out of memory"
```
MemoryError: Unable to allocate array
```

**Solutions:**
```bash
# Check system memory
free -h

# Reduce image processing batch size
export IMAGE_BATCH_SIZE=1

# Enable streaming data loading
export USE_STREAMING_LOADER=true

# Increase swap space (temporary solution)
sudo swapon --show
```

### 3. MiDaS Depth Extraction Issues

#### Error: "MiDaS model download failed"
```
OSError: Can't load tokenizer for 'Intel/dpt-large'
```

**Causes:**
- Network connectivity issues
- HuggingFace Hub access problems
- Insufficient disk space

**Solutions:**
```bash
# Check internet connectivity
curl -I https://huggingface.co

# Check disk space
df -h

# Manual model download
python -c "
from transformers import DPTImageProcessor, DPTForDepthEstimation
processor = DPTImageProcessor.from_pretrained('Intel/dpt-large')
model = DPTForDepthEstimation.from_pretrained('Intel/dpt-large')
print('MiDaS model downloaded successfully')
"

# Set HuggingFace cache directory
export HF_HOME=/path/to/large/disk/hf_cache
```

#### Error: "Depth map quality too low"
```
ValueError: Depth quality score 0.25 below threshold 0.5
```

**Causes:**
- Poor input image quality
- Inappropriate image for depth extraction
- MiDaS model limitations

**Solutions:**
```python
# Lower quality threshold
config = {
    "min_depth_quality": 0.2,  # Lower threshold
    "quality_threshold": 0.3
}

# Preprocess image for better depth extraction
from PIL import Image, ImageEnhance
import numpy as np

def enhance_image_for_depth(image_path):
    image = Image.open(image_path)
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.2)
    
    # Enhance sharpness
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.1)
    
    return image

# Try different MiDaS model
extractor = MiDaSDepthExtractor(model_name="Intel/dpt-hybrid-midas")
```

### 4. ControlNet Generation Issues

#### Error: "ControlNet model not found"
```
OSError: lllyasviel/sd-controlnet-depth does not appear to be a repository
```

**Solutions:**
```bash
# Check ControlNet model availability
python -c "
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
controlnet = ControlNetModel.from_pretrained('lllyasviel/sd-controlnet-depth')
print('ControlNet model loaded successfully')
"

# Use alternative ControlNet model
export CONTROLNET_MODEL="lllyasviel/control_v11f1p_sd15_depth"

# Download models manually
huggingface-cli download lllyasviel/sd-controlnet-depth
```

#### Error: "Generated image quality poor"
```
Warning: Generated image quality score 0.3 below threshold 0.7
```

**Solutions:**
```python
# Adjust generation parameters
generation_config = {
    "guidance_scale": 7.5,  # Increase for better prompt following
    "num_inference_steps": 50,  # Increase for better quality
    "controlnet_conditioning_scale": 1.2,  # Increase depth influence
    "strength": 0.8  # Control generation strength
}

# Improve prompts
positive_prompt = "photorealistic beach camera view, ocean waves breaking, natural lighting, high quality, detailed, sharp focus"
negative_prompt = "cartoon, anime, low quality, blurry, distorted, unrealistic, artificial"

# Use better base model
base_model = "runwayml/stable-diffusion-v1-5"  # or "stabilityai/stable-diffusion-2-1"
```

### 5. Training Issues

#### Error: "Training loss not decreasing"
```
Warning: Loss has not improved for 10 epochs
```

**Causes:**
- Learning rate too high or too low
- Poor data quality
- Model architecture issues
- Insufficient training data

**Solutions:**
```python
# Adjust learning rate
training_config = TrainingConfig(
    learning_rate=1e-4,  # Try different values: 1e-3, 1e-5
    scheduler='cosine',   # Use learning rate scheduling
    warmup_epochs=5
)

# Check data quality
from swellsight.training.data_quality_monitor import DataQualityMonitor
monitor = DataQualityMonitor()
quality_report = monitor.analyze_dataset("path/to/dataset")
print(quality_report)

# Increase dataset size
python swellsight/scripts/generate_training_data.py --samples 10000

# Adjust model architecture
model_config = ModelConfig(
    backbone='convnext_base',  # Try 'resnet50', 'efficientnet_b0'
    hidden_dim=512,            # Adjust hidden dimensions
    dropout_rate=0.2           # Add regularization
)
```

#### Error: "Gradient explosion"
```
RuntimeError: Function 'AddBackward0' returned nan values in its 0th output
```

**Solutions:**
```python
# Enable gradient clipping
training_config = TrainingConfig(
    gradient_clip_norm=1.0,  # Clip gradients
    learning_rate=1e-5       # Reduce learning rate
)

# Check for NaN in data
import torch
def check_for_nan(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
        return True
    return False

# Add gradient monitoring
def monitor_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm()
            if grad_norm > 10.0:
                print(f"Large gradient in {name}: {grad_norm}")
```

### 6. API Issues

#### Error: "API endpoint not responding"
```
ConnectionError: HTTPConnectionPool(host='localhost', port=8000)
```

**Solutions:**
```bash
# Check if API is running
curl http://localhost:8000/health

# Check process status
ps aux | grep python

# Check port availability
netstat -tlnp | grep 8000

# Restart API service
docker-compose restart swellsight-api

# Check logs
docker-compose logs swellsight-api
```

#### Error: "Request timeout"
```
TimeoutError: Request timed out after 30 seconds
```

**Solutions:**
```python
# Increase timeout settings
import requests
response = requests.post(
    "http://localhost:8000/predict",
    files={"image": image_data},
    timeout=120  # Increase timeout
)

# Enable async processing
@app.post("/predict_async")
async def predict_async(image: UploadFile):
    task_id = str(uuid.uuid4())
    # Queue prediction task
    prediction_queue.put({"task_id": task_id, "image": image})
    return {"task_id": task_id, "status": "queued"}

# Optimize model inference
model.eval()
with torch.no_grad():
    prediction = model(image_tensor)
```

### 7. Performance Issues

#### Error: "Slow inference speed"
```
Warning: Inference time 15.2s exceeds threshold 2.0s
```

**Solutions:**
```python
# Enable model quantization
import torch.quantization as quantization
model_int8 = quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Use TensorRT optimization (NVIDIA GPUs)
import torch_tensorrt
trt_model = torch_tensorrt.compile(
    model,
    inputs=[torch.randn(1, 3, 768, 768).cuda()],
    enabled_precisions={torch.float, torch.half}
)

# Enable batch processing
def batch_predict(images):
    batch_tensor = torch.stack(images)
    with torch.no_grad():
        predictions = model(batch_tensor)
    return predictions

# Use caching
from functools import lru_cache
@lru_cache(maxsize=1000)
def cached_predict(image_hash):
    return model.predict(image_hash)
```

### 8. Data Pipeline Issues

#### Error: "Dataset loading failed"
```
FileNotFoundError: Dataset metadata file not found
```

**Solutions:**
```bash
# Check dataset structure
ls -la data/
ls -la data/metadata/

# Regenerate metadata
python swellsight/scripts/generate_training_data.py --regenerate-metadata

# Verify dataset integrity
python -c "
from swellsight.data import DatasetManager
dm = DatasetManager('data/')
print(f'Dataset size: {len(dm)}')
print(f'Metadata files: {dm.list_metadata_files()}')
"
```

#### Error: "Data corruption detected"
```
ValueError: Invalid image data or corrupted file
```

**Solutions:**
```python
# Validate dataset
from swellsight.training.data_quality_monitor import DataQualityMonitor

monitor = DataQualityMonitor()
corrupted_files = monitor.find_corrupted_files("data/synthetic/")
print(f"Found {len(corrupted_files)} corrupted files")

# Remove corrupted files
for file_path in corrupted_files:
    os.remove(file_path)
    print(f"Removed corrupted file: {file_path}")

# Regenerate missing data
python swellsight/scripts/generate_training_data.py --replace-corrupted
```

## Debugging Tools

### 1. Model Debugging

```python
# Model summary and analysis
from swellsight.models import WaveAnalysisModel
from swellsight.config import ModelConfig

config = ModelConfig()
model = WaveAnalysisModel(config)

# Print model architecture
print(model)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Test forward pass
import torch
dummy_input = torch.randn(1, 3, 768, 768)
try:
    output = model(dummy_input)
    print("Forward pass successful")
    print(f"Output shapes: {[v.shape for v in output.values()]}")
except Exception as e:
    print(f"Forward pass failed: {e}")
```

### 2. Data Pipeline Debugging

```python
# Debug data loading
from swellsight.data import DatasetManager
import matplotlib.pyplot as plt

dm = DatasetManager("data/")
sample = dm[0]

print(f"Sample keys: {sample.keys()}")
print(f"Image shape: {sample['image'].shape}")
print(f"Labels: {sample['labels']}")

# Visualize sample
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(sample['image'].permute(1, 2, 0))
plt.title("Input Image")

if 'depth_map' in sample:
    plt.subplot(1, 2, 2)
    plt.imshow(sample['depth_map'], cmap='viridis')
    plt.title("Depth Map")

plt.show()
```

### 3. Performance Profiling

```python
# Profile inference performance
import time
import torch.profiler

def profile_inference(model, input_tensor):
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with torch.no_grad():
            output = model(input_tensor)
    
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    return output

# Memory profiling
def profile_memory():
    import psutil
    import torch
    
    process = psutil.Process()
    print(f"CPU Memory: {process.memory_info().rss / 1024**3:.2f} GB")
    
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
```

## Monitoring and Alerts

### 1. System Health Monitoring

```python
# Health check script
import psutil
import torch
import requests
from datetime import datetime

def system_health_check():
    health_status = {
        "timestamp": datetime.utcnow().isoformat(),
        "cpu_usage": psutil.cpu_percent(),
        "memory_usage": psutil.virtual_memory().percent,
        "disk_usage": psutil.disk_usage('/').percent,
        "gpu_available": torch.cuda.is_available()
    }
    
    if torch.cuda.is_available():
        health_status["gpu_memory_usage"] = (
            torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
        )
    
    # Check API health
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        health_status["api_status"] = response.status_code == 200
    except:
        health_status["api_status"] = False
    
    return health_status

# Run health check
health = system_health_check()
print(f"System Health: {health}")
```

### 2. Performance Monitoring

```python
# Performance metrics collection
import time
from collections import defaultdict

class PerformanceMonitor:
    def __init__(self):
        self.metrics = defaultdict(list)
    
    def record_inference_time(self, duration):
        self.metrics['inference_time'].append(duration)
    
    def record_memory_usage(self, usage):
        self.metrics['memory_usage'].append(usage)
    
    def get_stats(self):
        stats = {}
        for metric, values in self.metrics.items():
            if values:
                stats[metric] = {
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }
        return stats

# Usage
monitor = PerformanceMonitor()

start_time = time.time()
# ... run inference
inference_time = time.time() - start_time
monitor.record_inference_time(inference_time)

print(monitor.get_stats())
```

## Getting Help

### 1. Log Analysis

```bash
# Check application logs
tail -f logs/swellsight.log

# Filter error logs
grep -i error logs/swellsight.log

# Check Docker logs
docker-compose logs -f swellsight-api

# Check Kubernetes logs
kubectl logs -f deployment/swellsight-api
```

### 2. Support Channels

1. **Documentation**: Check the [FAQ](faq.md) and [Production Deployment Guide](production-deployment.md)
2. **Issue Tracking**: Create detailed bug reports with:
   - Error messages and stack traces
   - System configuration
   - Steps to reproduce
   - Expected vs actual behavior
3. **Performance Issues**: Include:
   - System specifications
   - Performance metrics
   - Profiling results
   - Configuration settings

### 3. Emergency Procedures

#### Service Outage
```bash
# Quick restart
docker-compose restart swellsight-api

# Rollback to previous version
kubectl rollout undo deployment/swellsight-api

# Scale down/up to refresh pods
kubectl scale deployment swellsight-api --replicas=0
kubectl scale deployment swellsight-api --replicas=3
```

#### Data Corruption
```bash
# Backup current state
cp -r data/ data_backup_$(date +%Y%m%d_%H%M%S)/

# Restore from backup
cp -r data_backup_latest/ data/

# Regenerate corrupted data
python swellsight/scripts/generate_training_data.py --repair-dataset
```

## Prevention Best Practices

1. **Regular Monitoring**: Set up automated health checks and alerts
2. **Backup Strategy**: Regular backups of models, data, and configurations
3. **Testing**: Comprehensive testing before deployment
4. **Documentation**: Keep troubleshooting logs and solutions updated
5. **Version Control**: Track all configuration and code changes
6. **Capacity Planning**: Monitor resource usage and plan for scaling

This troubleshooting guide should help resolve most common issues. For complex problems, gather detailed logs and system information before seeking support.