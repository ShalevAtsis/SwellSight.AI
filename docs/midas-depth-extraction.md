# MiDaS Depth Extraction Guide

This guide covers the setup, usage, and best practices for MiDaS depth extraction in the SwellSight Wave Analysis Model.

## Overview

The MiDaS (Monocular Depth Estimation) depth extractor uses HuggingFace's pre-trained models to extract depth maps from real beach camera images. These depth maps serve as structural guidance for ControlNet synthetic image generation and provide an alternative method for wave parameter estimation.

## Setup and Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch with CUDA support (recommended for GPU acceleration)
- HuggingFace Transformers library
- OpenCV for image processing

### Installation

The MiDaS depth extractor is included with the SwellSight installation:

```bash
pip install -e .
```

Or install dependencies directly:

```bash
pip install transformers torch torchvision opencv-python pillow
```

### Model Selection

SwellSight supports multiple MiDaS models:

- **Intel/dpt-large** (default): Best accuracy, slower inference
- **Intel/dpt-hybrid-midas**: Good balance of speed and accuracy
- **Intel/dpt-base**: Faster inference, lower accuracy

## Basic Usage

### Simple Depth Extraction

```python
from swellsight.data.midas_depth_extractor import MiDaSDepthExtractor

# Initialize extractor
extractor = MiDaSDepthExtractor(
    model_name="Intel/dpt-large",
    device="cuda"  # or "cpu"
)

# Extract depth from single image
result = extractor.extract_depth("path/to/beach_image.jpg")

print(f"Depth quality score: {result.depth_quality_score:.3f}")
print(f"Depth range: {result.processing_metadata['depth_range']}")

# Access depth map
depth_map = result.depth_map  # numpy array
```

### Batch Processing

```python
# Process multiple images
image_paths = [
    "data/real/images/beach1.jpg",
    "data/real/images/beach2.jpg",
    "data/real/images/beach3.jpg"
]

results = extractor.batch_extract(image_paths)

for result in results:
    print(f"Processed: {result.original_image_path}")
    print(f"Quality: {result.depth_quality_score:.3f}")
```

### With Storage System

```python
# Enable depth map storage
extractor = MiDaSDepthExtractor(
    model_name="Intel/dpt-large",
    storage_path="data/depth_maps"
)

# Extract and automatically store
result = extractor.extract_depth("beach_image.jpg", store_result=True)

# Retrieve stored depth maps
stored_maps = extractor.find_stored_depth_maps_by_image("beach_image.jpg")
```

## Configuration Options

### Model Configuration

```python
# High accuracy (slower)
extractor = MiDaSDepthExtractor(
    model_name="Intel/dpt-large",
    device="cuda"
)

# Balanced performance
extractor = MiDaSDepthExtractor(
    model_name="Intel/dpt-hybrid-midas",
    device="cuda"
)

# Fast inference (lower quality)
extractor = MiDaSDepthExtractor(
    model_name="Intel/dpt-base",
    device="cpu"
)
```

### Quality Validation Settings

The depth extractor includes built-in quality validation. You can customize the validation parameters:

```python
# Access quality validation
quality_score = extractor.validate_depth_quality(depth_map)

# Quality score interpretation:
# 0.0 - 0.3: Poor quality, may not be suitable for analysis
# 0.3 - 0.6: Moderate quality, usable with caution
# 0.6 - 0.8: Good quality, suitable for most applications
# 0.8 - 1.0: Excellent quality, ideal for analysis
```

## Depth Map Processing

### Understanding Depth Maps

MiDaS outputs depth maps where:
- **Lower values** = closer to camera (shallow water, beach)
- **Higher values** = farther from camera (deep water, horizon)
- **Range**: Typically 1-100 meters for beach scenes
- **Format**: 32-bit float numpy array

### Post-Processing Pipeline

The extractor applies several post-processing steps:

1. **Inversion**: MiDaS outputs inverse depth, converted to actual depth
2. **Normalization**: Scaled to 1-100 meter range for beach scenes
3. **Smoothing**: Gaussian blur to reduce noise
4. **Outlier Clipping**: Remove extreme values using percentiles

### Saving and Loading Depth Maps

```python
# Save depth map in different formats
extractor.save_depth_map(depth_map, "output/depth.npy", format="npy")
extractor.save_depth_map(depth_map, "output/depth.png", format="png")
extractor.save_depth_map(depth_map, "output/depth.tiff", format="tiff")

# Load depth map
loaded_depth = extractor.load_depth_map("output/depth.npy")
```

## Quality Assessment

### Quality Metrics

The depth extractor evaluates quality using four metrics:

1. **Variation Score**: Measures depth variation (good beach scenes have variation)
2. **Gradient Consistency**: Evaluates smooth transitions
3. **Spatial Coherence**: Checks neighboring pixel similarity
4. **Range Utilization**: Assesses dynamic range usage

### Quality Validation Example

```python
# Detailed quality assessment
result = extractor.extract_depth("beach_image.jpg")
metadata = result.analysis_metadata

print("Quality Indicators:")
for metric, value in metadata['quality_indicators'].items():
    print(f"  {metric}: {value:.3f}")

# Overall quality score
print(f"Overall Quality: {result.depth_quality_score:.3f}")
```

## Best Practices

### Image Selection

**Good candidates for depth extraction:**
- Clear beach camera views with visible water and shore
- Good contrast between water, foam, and beach
- Minimal occlusions (people, objects)
- Reasonable lighting conditions

**Avoid:**
- Heavily overexposed or underexposed images
- Images with significant motion blur
- Scenes with heavy fog or rain
- Images where water/beach boundary is unclear

### Performance Optimization

```python
# GPU acceleration (recommended)
extractor = MiDaSDepthExtractor(device="cuda")

# Batch processing for efficiency
results = extractor.batch_extract(image_paths)

# Enable caching for repeated processing
extractor = MiDaSDepthExtractor(storage_path="cache/depth_maps")
```

### Memory Management

```python
# For large batches, process in chunks
def process_large_batch(image_paths, chunk_size=10):
    results = []
    for i in range(0, len(image_paths), chunk_size):
        chunk = image_paths[i:i+chunk_size]
        chunk_results = extractor.batch_extract(chunk)
        results.extend(chunk_results)
        
        # Optional: clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results
```

## Integration with Pipeline

### End-to-End Pipeline Usage

```python
from swellsight.scripts.end_to_end_pipeline import EndToEndPipeline, PipelineConfig

# Configure pipeline with MiDaS settings
config = PipelineConfig()
config.config.update({
    'midas_model': 'Intel/dpt-large',
    'min_depth_quality': 0.4,
    'real_data_path': 'data/real'
})

# Run complete pipeline
pipeline = EndToEndPipeline(config)
results = pipeline.run()
```

### With ControlNet Generation

```python
from swellsight.data.controlnet_generator import ControlNetSyntheticGenerator

# Extract depth maps
depth_results = extractor.batch_extract(image_paths)

# Generate synthetic images using depth maps
generator = ControlNetSyntheticGenerator()
for depth_result in depth_results:
    if depth_result.depth_quality_score > 0.4:  # Quality threshold
        synthetic_result = generator.generate_synthetic_image(
            depth_result.depth_map,
            augmentation_params
        )
```

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```python
# Solution: Use CPU or smaller model
extractor = MiDaSDepthExtractor(device="cpu")
# or
extractor = MiDaSDepthExtractor(model_name="Intel/dpt-base")
```

**2. Poor Depth Quality**
```python
# Check quality score
if result.depth_quality_score < 0.3:
    print("Poor depth quality, consider:")
    print("- Different image")
    print("- Different MiDaS model")
    print("- Image preprocessing")
```

**3. Model Loading Errors**
```bash
# Ensure HuggingFace cache is accessible
export HF_HOME=/path/to/cache
# or clear cache
rm -rf ~/.cache/huggingface/
```

### Performance Issues

**Slow inference:**
- Use GPU acceleration with CUDA
- Consider smaller model (dpt-base vs dpt-large)
- Process images in batches
- Resize large images before processing

**Memory issues:**
- Process images in smaller batches
- Clear GPU cache between batches
- Use CPU for very large images

## Advanced Usage

### Custom Quality Validation

```python
class CustomDepthExtractor(MiDaSDepthExtractor):
    def validate_depth_quality(self, depth_map):
        # Custom quality validation logic
        custom_score = your_quality_function(depth_map)
        return custom_score

extractor = CustomDepthExtractor()
```

### Integration with Real Data Labels

```python
from swellsight.data.real_data_loader import RealDataLoader

# Load real data with labels
real_loader = RealDataLoader("data/real")
labeled_samples = real_loader.load_labeled_samples()

# Extract depth maps for labeled data
for sample in labeled_samples:
    depth_result = extractor.extract_depth(sample.image_path)
    
    # Compare depth-based estimates with manual labels
    estimated_height = depth_analyzer.estimate_wave_height(depth_result.depth_map)
    actual_height = sample.labels['wave_height_m']
    
    print(f"Estimated: {estimated_height[0]:.2f}m, Actual: {actual_height:.2f}m")
```

## API Reference

### MiDaSDepthExtractor Class

#### Constructor
```python
MiDaSDepthExtractor(
    model_name: str = "Intel/dpt-large",
    device: Optional[str] = None,
    storage_path: Optional[str] = None
)
```

#### Methods

**extract_depth(image_path, store_result=True)**
- Extract depth map from single image
- Returns: `DepthExtractionResult`

**batch_extract(image_paths)**
- Extract depth maps from multiple images
- Returns: `List[DepthExtractionResult]`

**validate_depth_quality(depth_map)**
- Validate depth map quality
- Returns: `float` (0.0-1.0)

**save_depth_map(depth_map, output_path, format)**
- Save depth map to file
- Formats: 'npy', 'png', 'tiff'

**load_depth_map(depth_path)**
- Load depth map from file
- Returns: `np.ndarray`

### DepthExtractionResult Class

#### Attributes
- `depth_map`: numpy array with depth values
- `original_image_path`: path to source image
- `depth_quality_score`: quality score (0.0-1.0)
- `processing_metadata`: processing information and parameters

## Examples

See the `examples/` directory for complete examples:
- `examples/basic_depth_extraction.py`
- `examples/batch_processing.py`
- `examples/quality_validation.py`
- `examples/pipeline_integration.py`