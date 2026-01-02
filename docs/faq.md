# Frequently Asked Questions (FAQ)

## General Questions

### Q: What is SwellSight?
**A:** SwellSight is a multi-task deep learning system that extracts objective physical wave parameters from beach camera images. It can predict wave height, breaking type, and wave direction from single images using a combination of MiDaS depth extraction, ControlNet synthetic data generation, and a ConvNeXt-based neural network.

### Q: What wave parameters can SwellSight predict?
**A:** SwellSight predicts three main wave parameters:
- **Wave Height**: Continuous values in meters (0.3-4.0m range)
- **Wave Type**: Classification into 4 categories (A-frame, closeout, beach break, point break)
- **Wave Direction**: Classification into 3 categories (left, right, both)

### Q: How accurate is SwellSight?
**A:** Performance varies by parameter and conditions:
- **Height**: ±0.3 meters MAE for waves 0.5-3.0m
- **Wave Type**: ~75% accuracy on real beach images
- **Direction**: ~80% accuracy on real beach images

Accuracy is higher on synthetic validation data and may vary based on image quality, lighting conditions, and wave complexity.

## Technical Questions

### Q: What image formats does SwellSight support?
**A:** SwellSight supports common image formats:
- **JPEG** (.jpg, .jpeg) - Recommended for beach camera images
- **PNG** (.png) - Supported with transparency handling
- **Input Size**: Images are automatically resized to 768x768 pixels
- **Color Space**: RGB color images (3 channels)

### Q: What are the system requirements?
**A:** Minimum requirements:
- **CPU**: 4+ cores, 8+ cores recommended for production
- **Memory**: 8GB RAM minimum, 16GB+ recommended
- **Storage**: 50GB+ for models and data
- **GPU**: Optional but recommended (8GB+ VRAM for ControlNet generation)
- **OS**: Linux (Ubuntu 20.04+), macOS, Windows with WSL2

### Q: Can SwellSight run without a GPU?
**A:** Yes, SwellSight can run on CPU-only systems:
- **Inference**: Works well on CPU with quantized models
- **Training**: Possible but significantly slower
- **MiDaS**: Runs on CPU with reasonable performance
- **ControlNet**: CPU generation is very slow; consider using pre-generated synthetic data

### Q: How long does inference take?
**A:** Inference times vary by hardware:
- **CPU**: 1-5 seconds per image
- **GPU**: 0.2-1 seconds per image
- **Batch Processing**: More efficient for multiple images
- **Quantized Models**: 2-3x faster on CPU

## Data and Training Questions

### Q: What training data does SwellSight use?
**A:** SwellSight uses a hybrid approach:
- **Synthetic Data**: Generated using MiDaS depth extraction + ControlNet
- **Real Data**: Beach camera images with manual labels (validation only)
- **Augmentation**: 10 comprehensive categories of scene variations
- **Dataset Size**: Typically 10,000+ synthetic samples for training

### Q: Can I train SwellSight on my own data?
**A:** Yes, you can customize training:
```python
# Add your real images to data/real/images/
# Add corresponding labels to data/real/labels/labels.json
# Generate synthetic variants
python swellsight/scripts/generate_training_data.py --use-real-images --samples 5000

# Train with custom data
python swellsight/scripts/train_end_to_end.py --config custom_config.json
```

### Q: How do I improve model accuracy?
**A:** Several strategies can improve accuracy:
1. **More Training Data**: Generate larger synthetic datasets
2. **Better Real Data**: Use high-quality beach camera images for validation
3. **Hyperparameter Tuning**: Adjust learning rate, batch size, model architecture
4. **Data Quality**: Ensure good lighting, clear wave visibility
5. **Domain Adaptation**: Fine-tune on real data from your specific camera setup

### Q: What is the augmentation parameter system?
**A:** The system varies 10 categories of beach scene attributes:
1. **Camera View Geometry**: Height, tilt, FOV, distance, offset
2. **Wave Field Structure**: Height, wavelength, period, spread, fronts
3. **Breaking Behavior**: Type, intensity, sharpness, foam coverage
4. **Shore Interaction**: Slope, run-up, backwash, reflectivity, curvature
5. **Water Surface Texture**: Roughness, ripples, streaks, highlights, micro-foam
6. **Lighting and Sun Position**: Elevation, azimuth, intensity, softness, glare
7. **Atmospheric Conditions**: Haze, fog, humidity, clarity, attenuation
8. **Weather State**: Clouds, type, rain, streaks, storminess
9. **Optical/Sensor Artifacts**: Distortion, blur, noise, compression, aberration
10. **Scene Occlusions/Noise**: People, surfboards, birds, spray, foreground blur

## Deployment Questions

### Q: How do I deploy SwellSight in production?
**A:** Follow the [Production Deployment Guide](production-deployment.md):
1. **Docker**: Use provided Dockerfile for containerization
2. **Kubernetes**: Apply provided manifests for orchestration
3. **API**: RESTful API with OpenAPI specification
4. **Monitoring**: Prometheus metrics and health checks
5. **Scaling**: Horizontal pod autoscaling based on load

### Q: Can SwellSight handle multiple concurrent requests?
**A:** Yes, SwellSight supports concurrent processing:
- **Async API**: FastAPI with async request handling
- **Batch Processing**: Multiple images in single forward pass
- **Load Balancing**: Multiple replicas behind load balancer
- **Caching**: Redis caching for repeated requests
- **Queue System**: Background processing for high loads

### Q: How do I monitor SwellSight in production?
**A:** Comprehensive monitoring is available:
```python
# Health endpoints
GET /health      # Basic health check
GET /ready       # Readiness probe
GET /metrics     # Prometheus metrics

# Key metrics to monitor
- Request rate and response time
- Model accuracy and confidence scores
- Memory and CPU usage
- Error rates and types
- Queue depth and processing time
```

## MiDaS and ControlNet Questions

### Q: What is MiDaS and why is it used?
**A:** MiDaS (Monocular Depth Estimation) extracts depth information from single images:
- **Purpose**: Provides 3D structure information from 2D beach images
- **Model**: Intel DPT-Large from HuggingFace
- **Usage**: Creates depth maps that guide ControlNet synthetic generation
- **Benefits**: Preserves wave structure and spatial relationships

### Q: What is ControlNet and how does it work?
**A:** ControlNet generates synthetic images conditioned on depth maps:
- **Base Model**: Stable Diffusion with depth conditioning
- **Input**: MiDaS depth maps + text prompts + augmentation parameters
- **Output**: Photorealistic beach camera images with controlled wave characteristics
- **Benefits**: Creates diverse training data while preserving ground truth labels

### Q: Can I use different MiDaS or ControlNet models?
**A:** Yes, models are configurable:
```python
# MiDaS alternatives
"Intel/dpt-large"        # Best quality, slower
"Intel/dpt-hybrid-midas" # Good balance
"Intel/dpt-swinv2-tiny-256" # Faster, lower quality

# ControlNet alternatives
"lllyasviel/sd-controlnet-depth"     # Standard depth ControlNet
"lllyasviel/control_v11f1p_sd15_depth" # Improved version
"thibaud/controlnet-sd21-depth-diffusers" # SD 2.1 based
```

### Q: Why is ControlNet generation slow?
**A:** ControlNet is computationally intensive:
- **GPU Required**: CPU generation is extremely slow
- **Memory Usage**: Requires 8GB+ VRAM for optimal performance
- **Inference Steps**: 20-50 steps per image (adjustable)
- **Solutions**: Use GPU, reduce inference steps, batch processing, pre-generate data

## Troubleshooting Questions

### Q: SwellSight is running out of memory, what should I do?
**A:** Several memory optimization strategies:
```bash
# Reduce batch size
export BATCH_SIZE=1

# Enable memory efficient attention
export ENABLE_MEMORY_EFFICIENT_ATTENTION=true

# Use CPU instead of GPU
export DEVICE=cpu

# Enable model quantization
export USE_QUANTIZED_MODEL=true
```

### Q: Model predictions seem unrealistic, how to debug?
**A:** Debug prediction quality:
```python
# Check input image quality
from PIL import Image
image = Image.open("test_image.jpg")
print(f"Image size: {image.size}")
print(f"Image mode: {image.mode}")

# Visualize preprocessing
from swellsight.inference import InferenceEngine
engine = InferenceEngine(model, config)
preprocessed = engine.preprocess_image("test_image.jpg")

# Check model confidence scores
prediction = engine.predict("test_image.jpg")
print(f"Confidence scores: {prediction.confidence_scores}")

# Compare with depth-based analysis
from swellsight.data.depth_analyzer import DepthAnalyzer
analyzer = DepthAnalyzer()
depth_prediction = analyzer.analyze_image("test_image.jpg")
```

### Q: Training is not converging, what's wrong?
**A:** Common training issues and solutions:
1. **Learning Rate**: Try 1e-4, 1e-5, or use learning rate scheduling
2. **Data Quality**: Check for corrupted images or incorrect labels
3. **Model Architecture**: Verify configuration matches intended design
4. **Loss Weights**: Adjust multi-task loss balancing
5. **Gradient Issues**: Enable gradient clipping, check for NaN values

### Q: API is returning errors, how to diagnose?
**A:** API debugging steps:
```bash
# Check API health
curl http://localhost:8000/health

# Check logs
docker-compose logs swellsight-api

# Test with simple image
curl -X POST -F "image=@test.jpg" http://localhost:8000/predict

# Validate input format
file test.jpg  # Should show JPEG image data
```

## Performance Questions

### Q: How can I make inference faster?
**A:** Performance optimization strategies:
1. **Model Quantization**: Use INT8 quantized models for CPU
2. **Batch Processing**: Process multiple images together
3. **GPU Acceleration**: Use CUDA-enabled GPU
4. **Caching**: Cache results for repeated requests
5. **TensorRT**: Use TensorRT optimization on NVIDIA GPUs

### Q: How much storage does SwellSight need?
**A:** Storage requirements:
- **Models**: ~2GB for all model weights
- **Training Data**: 10-100GB depending on dataset size
- **Depth Maps**: ~50MB per 1000 images (compressed)
- **Logs**: 1-10GB depending on retention policy
- **Cache**: 1-10GB for prediction caching

### Q: Can SwellSight scale horizontally?
**A:** Yes, SwellSight is designed for horizontal scaling:
- **Stateless API**: No shared state between instances
- **Load Balancing**: Multiple replicas behind load balancer
- **Auto Scaling**: Kubernetes HPA based on CPU/memory
- **Caching**: Shared Redis cache across instances
- **Model Storage**: Shared model registry for consistency

## Integration Questions

### Q: How do I integrate SwellSight with my application?
**A:** Integration options:
```python
# REST API (recommended)
import requests
response = requests.post(
    "http://swellsight-api/predict",
    files={"image": open("beach.jpg", "rb")}
)
prediction = response.json()

# Python SDK
from swellsight import SwellSightClient
client = SwellSightClient("http://swellsight-api")
prediction = client.predict("beach.jpg")

# Batch processing
predictions = client.batch_predict(["img1.jpg", "img2.jpg", "img3.jpg"])
```

### Q: What data format does the API return?
**A:** API returns structured JSON:
```json
{
  "height_meters": 1.85,
  "wave_type": "A_FRAME",
  "direction": "RIGHT",
  "confidence_scores": {
    "height": 0.87,
    "wave_type": 0.92,
    "direction": 0.78
  },
  "processing_time": 1.23,
  "model_version": "1.0.0"
}
```

### Q: Can I get raw model outputs instead of processed predictions?
**A:** Yes, use the debug endpoint:
```python
# Get raw model outputs
response = requests.post(
    "http://swellsight-api/predict_raw",
    files={"image": open("beach.jpg", "rb")}
)

raw_output = response.json()
# Contains raw logits, probabilities, and intermediate features
```

## Licensing and Usage Questions

### Q: Can I use SwellSight commercially?
**A:** Check the project license for commercial usage terms. The system uses several open-source components with different licenses:
- **MiDaS**: MIT License
- **ControlNet**: Apache 2.0 License
- **Stable Diffusion**: CreativeML Open RAIL-M License
- **SwellSight Code**: Check repository license

### Q: Can I modify SwellSight for my specific use case?
**A:** Yes, SwellSight is designed to be extensible:
- **Custom Models**: Replace backbone architecture
- **New Parameters**: Add additional wave characteristics
- **Different Domains**: Adapt for other water body types
- **Custom Augmentation**: Add domain-specific augmentation categories

### Q: How do I contribute to SwellSight development?
**A:** Contributions are welcome:
1. **Bug Reports**: Use GitHub issues with detailed reproduction steps
2. **Feature Requests**: Describe use case and proposed implementation
3. **Code Contributions**: Follow coding standards and include tests
4. **Documentation**: Improve guides, examples, and API documentation
5. **Testing**: Add test cases and validation scenarios

## Getting More Help

### Q: Where can I find more detailed documentation?
**A:** Additional resources:
- [Production Deployment Guide](production-deployment.md)
- [Troubleshooting Guide](troubleshooting.md)
- [MiDaS Depth Extraction](midas-depth-extraction.md)
- [ControlNet Synthetic Generation](controlnet-synthetic-generation.md)
- [Augmentation Parameters](augmentation-parameters.md)
- [Depth-Based Analysis](depth-based-analysis.md)

### Q: How do I report bugs or request features?
**A:** Use the appropriate channels:
1. **Bugs**: Create GitHub issue with:
   - Error messages and stack traces
   - Steps to reproduce
   - System configuration
   - Expected vs actual behavior

2. **Feature Requests**: Create GitHub issue with:
   - Use case description
   - Proposed solution
   - Alternative approaches considered
   - Impact assessment

### Q: Is there a community or support forum?
**A:** Check the project repository for:
- **GitHub Discussions**: Community Q&A and feature discussions
- **Issue Tracker**: Bug reports and feature requests
- **Documentation**: Comprehensive guides and examples
- **Code Examples**: Sample implementations and integrations

### Q: How often is SwellSight updated?
**A:** Update frequency depends on:
- **Bug Fixes**: Released as needed
- **Feature Updates**: Regular releases with new capabilities
- **Model Improvements**: Updated models with better accuracy
- **Security Updates**: Prompt updates for security issues
- **Dependency Updates**: Regular updates for underlying libraries

For the latest information, check the project repository and release notes.